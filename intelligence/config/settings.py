"""
Application Settings
=====================
Pydantic Settings — reads from environment variables and .env file.
All services read from this single source of truth.

Usage:
    from config.settings import get_settings
    settings = get_settings()
    print(settings.slm_provider)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root (intelligence/)
_INTEL_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_INTEL_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────────────────────────
    app_name: str = "PRAJNA Intelligence API"
    app_version: str = "1.0.0"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # ── Server ────────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8001
    workers: int = 1
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:4000", "http://localhost:8080", "http://localhost:8765", "http://localhost:8502", "http://127.0.0.1:8765", "http://127.0.0.1:4000"],
    )

    # ── SLM Provider ──────────────────────────────────────────────────────────
    slm_provider: Literal["ollama", "huggingface", "openai_compatible", "mock"] = "mock"

    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral:7b-instruct"
    ollama_timeout: int = 120

    # HuggingFace settings
    hf_model_id: str = "microsoft/phi-3-mini-4k-instruct"
    hf_device: str = "cpu"           # "cpu" | "cuda" | "mps"
    hf_load_in_4bit: bool = False
    hf_adapter_path: Optional[str] = None   # LoRA / PEFT adapter

    # OpenAI-compatible settings (OpenAI, Together.ai, Groq, vLLM, LM Studio)
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = "sk-placeholder"
    openai_model: str = "gpt-3.5-turbo"
    openai_timeout: int = 60

    # SLM generation parameters
    slm_max_tokens: int = 1024
    slm_temperature: float = 0.2     # Low for grounded, factual responses
    slm_top_p: float = 0.9

    # ── Prediction Adapter ────────────────────────────────────────────────────
    prediction_adapter_mode: Literal["local", "http"] = "local"
    prediction_engine_url: str = "http://localhost:8000"   # PRAJNA API (if http mode)
    prediction_cache_ttl_sec: int = 3600

    # Path to PRAJNA engine (used in local mode)
    prajna_engine_path: Optional[str] = None   # defaults to auto-discovery

    # ── RAG / Vector Store ────────────────────────────────────────────────────
    rag_enabled: bool = True
    chroma_host: str = "localhost"
    chroma_port: int = 8002
    chroma_collection: str = "prajna_intelligence"

    # Embedding model (must match what PRAJNA SLM uses)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"

    # Retrieval settings
    rag_top_k: int = 5
    rag_min_relevance: float = 0.35

    # ── Data Paths ────────────────────────────────────────────────────────────
    data_root: Path = Field(default=_INTEL_ROOT.parent / "data")
    student_data_dir: Path = Field(default=_INTEL_ROOT.parent / "data" / "student_data")
    syllabus_path: Optional[Path] = None   # auto-detected from data_root

    # ── Cache / Redis (optional) ───────────────────────────────────────────────
    redis_enabled: bool = False
    redis_url: str = "redis://localhost:6379/0"
    insight_cache_ttl_sec: int = 1800   # 30 min insight cache

    # ── API Limits ────────────────────────────────────────────────────────────
    max_top_n: int = 50
    max_revision_subjects: int = 10
    request_timeout_sec: int = 90

    # ── Evaluation ────────────────────────────────────────────────────────────
    eval_output_dir: Path = Field(default=_INTEL_ROOT / "docs" / "eval_results")

    # ── Validators ────────────────────────────────────────────────────────────
    @field_validator("slm_temperature")
    @classmethod
    def clamp_temperature(cls, v: float) -> float:
        return max(0.0, min(2.0, v))

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    # ── Derived helpers ────────────────────────────────────────────────────────
    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def effective_data_root(self) -> Path:
        return self.data_root.resolve()

    def get_slm_kwargs(self) -> dict:
        """Return provider-specific kwargs for create_provider()."""
        if self.slm_provider == "ollama":
            return {
                "base_url": self.ollama_base_url,
                "model":    self.ollama_model,
                "timeout":  self.ollama_timeout,
                "max_tokens": self.slm_max_tokens,
                "temperature": self.slm_temperature,
            }
        elif self.slm_provider == "huggingface":
            return {
                "model_id":    self.hf_model_id,
                "device":      self.hf_device,
                "load_in_4bit": self.hf_load_in_4bit,
                "adapter_path": self.hf_adapter_path,
                "max_new_tokens": self.slm_max_tokens,
                "temperature": self.slm_temperature,
            }
        elif self.slm_provider == "openai_compatible":
            return {
                "base_url": self.openai_base_url,
                "api_key":  self.openai_api_key,
                "model":    self.openai_model,
                "timeout":  self.openai_timeout,
                "max_tokens": self.slm_max_tokens,
                "temperature": self.slm_temperature,
            }
        return {}   # mock — no kwargs needed


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached Settings singleton.
    Call get_settings.cache_clear() in tests to reset.
    """
    return Settings()
