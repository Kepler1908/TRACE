"""
Configuration management for trace_rag.

Pydantic Settings with env variable support for LLM providers, logging, and application settings.
"""

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """LLM provider configuration using official provider SDKs."""

    provider: Literal["openai", "anthropic", "google", "deepseek", "openrouter", "llama_cpp", "vllm", "sglang"] = Field(
        default="deepseek",
        validation_alias=AliasChoices("DEFAULT_LLM_PROVIDER", "LLM_PROVIDER"),
    )
    model: str = Field(
        default="deepseek-chat",
        validation_alias=AliasChoices("DEFAULT_LLM_MODEL", "LLM_MODEL"),
    )

    # API Keys (loaded from environment)
    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "LLM_OPENAI_API_KEY"),
    )
    anthropic_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "LLM_ANTHROPIC_API_KEY"),
    )
    google_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GOOGLE_API_KEY", "LLM_GOOGLE_API_KEY"),
    )
    deepseek_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("DEEPSEEK_API_KEY", "LLM_DEEPSEEK_API_KEY"),
    )
    openrouter_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENROUTER_API_KEY", "LLM_OPENROUTER_API_KEY"),
    )

    # llama.cpp (local OpenAI-compatible server)
    llama_cpp_base_url: str = Field(
        default="http://localhost:8080/v1",
        validation_alias=AliasChoices("LLAMA_CPP_BASE_URL", "LLM_LLAMA_CPP_BASE_URL"),
    )
    llama_cpp_api_key: str = "unused"

    # vLLM (local OpenAI-compatible server)
    vllm_base_url: str = Field(
        default="http://localhost:8000/v1",
        validation_alias=AliasChoices("VLLM_BASE_URL", "LLM_VLLM_BASE_URL"),
    )
    vllm_api_key: str = "unused"

    # SGLang (local OpenAI-compatible server)
    sglang_base_url: str = Field(
        default="http://localhost:30000/v1",
        validation_alias=AliasChoices("SGLANG_BASE_URL", "LLM_SGLANG_BASE_URL"),
    )
    sglang_api_key: str = "unused"

    # Provider base URLs
    openai_base_url: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    anthropic_base_url: str | None = None

    # Request settings
    temperature: float = 0.1
    max_tokens: int = 8192
    timeout: int = 120
    max_retries: int = 3

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="LLM_",
        populate_by_name=True,
        extra="ignore",
    )


class EmbeddingSettings(BaseSettings):
    """Embedding model configuration."""

    model: str = Field(
        default="Qwen/Qwen3-Embedding-8B",
        validation_alias=AliasChoices("EMBEDDING_MODEL", "DEFAULT_EMBEDDING_MODEL"),
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["json", "console"] = "console"

    model_config = SettingsConfigDict(env_prefix="LOG_")


class Settings(BaseSettings):
    """Main application settings."""

    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    debug: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
