"""Configuration for NoteMind.

Defaults live in ``config.yaml``; this module turns that file into typed,
validated, immutable dataclasses. Immutability matters: the Streamlit UI lets
the user change settings, and instead of mutating a shared object it builds a
new config with :meth:`AppConfig.with_overrides`, so two browser sessions can
never change each other's settings.

API keys are never stored in config. They are read from environment variables
(optionally loaded from a ``.env`` file) by :func:`get_api_key`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# src/notemind/config.py -> parents[2] is the project root (editable install).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config.yaml"

VALID_MODES: tuple[str, ...] = ("heap", "faiss", "bm25", "hybrid")
VALID_DENSE_BACKENDS: tuple[str, ...] = ("heap", "faiss")
VALID_PROVIDERS: tuple[str, ...] = ("anthropic", "gemini", "groq")
# None = don't send a thinking setting (use the model's default).
VALID_THINKING_LEVELS: tuple[str | None, ...] = (None, "minimal", "low", "medium", "high")

# Environment variables checked for each provider, in order of preference.
API_KEY_ENV_VARS: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "groq": ("GROQ_API_KEY",),
}

# all-MiniLM-L6-v2 truncates at 256 word-piece tokens, roughly 1,000-1,200
# characters of English. Beyond this, the end of each chunk is never embedded.
MINILM_SAFE_CHUNK_CHARS: int = 1000


class ConfigError(ValueError):
    """Raised when config.yaml is missing, malformed or has invalid values."""


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem locations. All paths are absolute after loading."""

    data_dir: Path
    index_dir: Path
    eval_set: Path
    eval_results_dir: Path
    demo_dir: Path


@dataclass(frozen=True)
class ChunkingConfig:
    """How page text is split into overlapping chunks (sizes in characters)."""

    chunk_size: int = 800
    chunk_overlap: int = 150
    min_chunk_chars: int = 40

    def __post_init__(self) -> None:
        if self.chunk_size < 100:
            raise ConfigError(f"chunk_size must be >= 100, got {self.chunk_size}")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ConfigError(
                f"chunk_overlap must be in [0, chunk_size), got {self.chunk_overlap} "
                f"with chunk_size={self.chunk_size}"
            )
        if self.min_chunk_chars < 0:
            raise ConfigError("min_chunk_chars must be >= 0")
        if self.chunk_size > MINILM_SAFE_CHUNK_CHARS:
            logger.warning(
                "chunk_size=%d exceeds ~%d chars; all-MiniLM-L6-v2 truncates at 256 "
                "tokens, so the end of long chunks will not be embedded.",
                self.chunk_size,
                MINILM_SAFE_CHUNK_CHARS,
            )


@dataclass(frozen=True)
class EmbeddingConfig:
    """Sentence-transformers model settings."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ConfigError("embedding.batch_size must be >= 1")


@dataclass(frozen=True)
class RetrievalConfig:
    """Which retriever to use and how many results to keep."""

    mode: str = "hybrid"
    dense_backend: str = "faiss"
    top_k: int = 5
    candidate_k: int = 20
    rrf_k: int = 60
    min_similarity: float = 0.25

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise ConfigError(f"retrieval.mode must be one of {VALID_MODES}, got {self.mode!r}")
        if self.dense_backend not in VALID_DENSE_BACKENDS:
            raise ConfigError(
                f"retrieval.dense_backend must be one of {VALID_DENSE_BACKENDS}, "
                f"got {self.dense_backend!r}"
            )
        if self.top_k < 1:
            raise ConfigError("retrieval.top_k must be >= 1")
        if self.candidate_k < self.top_k:
            raise ConfigError("retrieval.candidate_k must be >= top_k")
        if self.rrf_k < 1:
            raise ConfigError("retrieval.rrf_k must be >= 1")
        if not -1.0 <= self.min_similarity <= 1.0:
            raise ConfigError("retrieval.min_similarity must be a cosine value in [-1, 1]")


@dataclass(frozen=True)
class GenerationConfig:
    """LLM provider settings. The API key is NOT stored here."""

    provider: str = "gemini"
    models: dict[str, str] = field(
        default_factory=lambda: {
            "anthropic": "claude-haiku-5-5",
            "gemini": "gemini-3.5-flash",
            "groq": "openai/gpt-oss-20b",
        }
    )
    temperature: float = 0.1
    max_tokens: int = 1024
    max_retries: int = 3
    timeout_seconds: float = 60.0
    max_context_chars: int = 12000
    gemini_thinking_level: str | None = "low"

    def __post_init__(self) -> None:
        if self.provider not in VALID_PROVIDERS:
            raise ConfigError(
                f"generation.provider must be one of {VALID_PROVIDERS}, got {self.provider!r}"
            )
        if self.provider not in self.models or not self.models[self.provider]:
            raise ConfigError(f"generation.models has no model ID for provider {self.provider!r}")
        if not 0.0 <= self.temperature <= 2.0:
            raise ConfigError("generation.temperature must be in [0, 2]")
        if self.max_tokens < 64:
            raise ConfigError("generation.max_tokens must be >= 64")
        if self.max_retries < 0:
            raise ConfigError("generation.max_retries must be >= 0")
        if self.timeout_seconds <= 0:
            raise ConfigError("generation.timeout_seconds must be > 0")
        if self.max_context_chars < 1000:
            raise ConfigError("generation.max_context_chars must be >= 1000")
        if self.gemini_thinking_level not in VALID_THINKING_LEVELS:
            raise ConfigError(
                f"generation.gemini_thinking_level must be one of {VALID_THINKING_LEVELS}"
            )

    @property
    def model(self) -> str:
        """Model ID for the currently selected provider."""
        return self.models[self.provider]


@dataclass(frozen=True)
class QuizConfig:
    """Quiz generation limits."""

    default_num_questions: int = 5
    max_num_questions: int = 15
    max_source_chars: int = 15000

    def __post_init__(self) -> None:
        if not 1 <= self.default_num_questions <= self.max_num_questions:
            raise ConfigError("quiz.default_num_questions must be in [1, max_num_questions]")
        if self.max_source_chars < 1000:
            raise ConfigError("quiz.max_source_chars must be >= 1000")


@dataclass(frozen=True)
class AppConfig:
    """Top-level configuration object passed to every component."""

    paths: PathsConfig
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    quiz: QuizConfig = field(default_factory=QuizConfig)
    log_level: str = "INFO"

    def with_overrides(self, **sections: dict[str, Any]) -> "AppConfig":
        """Return a new config with some section values replaced.

        Example::

            cfg2 = cfg.with_overrides(chunking={"chunk_size": 500},
                                      retrieval={"mode": "bm25", "top_k": 3})

        The new section objects are re-validated, so invalid overrides raise
        :class:`ConfigError` instead of silently producing a broken pipeline.
        """
        changes: dict[str, Any] = {}
        for section_name, values in sections.items():
            current = getattr(self, section_name, None)
            if current is None or not hasattr(current, "__dataclass_fields__"):
                raise ConfigError(f"Unknown config section: {section_name!r}")
            _reject_unknown_keys(type(current), values, section_name)
            changes[section_name] = replace(current, **values)
        return replace(self, **changes)


def _reject_unknown_keys(cls: type, data: dict[str, Any], section: str) -> None:
    """Fail loudly on typos such as ``chunk_sise`` instead of ignoring them."""
    allowed = {f.name for f in fields(cls)}
    unknown = set(data) - allowed
    if unknown:
        raise ConfigError(
            f"Unknown key(s) in '{section}': {sorted(unknown)}. Allowed: {sorted(allowed)}"
        )


def _build_section(cls: type, data: dict[str, Any] | None, section: str) -> Any:
    """Instantiate one config dataclass from a YAML mapping."""
    data = data or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Section '{section}' must be a mapping, got {type(data).__name__}")
    _reject_unknown_keys(cls, data, section)
    try:
        return cls(**data)
    except TypeError as exc:  # wrong value types, e.g. a list where an int belongs
        raise ConfigError(f"Invalid values in '{section}': {exc}") from exc


def _build_paths(data: dict[str, Any] | None, base_dir: Path) -> PathsConfig:
    """Resolve path strings relative to the config file's folder."""
    defaults = {
        "data_dir": "data/sample_pdfs",
        "index_dir": "indexes",
        "eval_set": "eval/eval_set.json",
        "eval_results_dir": "eval/results",
        "demo_dir": "assets/demo",
    }
    merged = {**defaults, **(data or {})}
    _reject_unknown_keys(PathsConfig, merged, "paths")
    resolved = {
        key: (Path(value) if Path(value).is_absolute() else (base_dir / value)).resolve()
        for key, value in merged.items()
    }
    return PathsConfig(**resolved)


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load and validate configuration.

    Resolution order for the file: the ``path`` argument, then the
    ``NOTEMIND_CONFIG`` environment variable, then ``<project>/config.yaml``.

    Args:
        path: Optional explicit path to a YAML config file.

    Returns:
        A fully validated, immutable :class:`AppConfig`.

    Raises:
        ConfigError: If the file is missing, is not valid YAML, or has bad values.
    """
    config_path = Path(path or os.environ.get("NOTEMIND_CONFIG") or DEFAULT_CONFIG_PATH)
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} must contain a mapping at the top level")

    known = {"paths", "chunking", "embedding", "retrieval", "generation", "quiz", "logging"}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"Unknown top-level section(s): {sorted(unknown)}")

    base_dir = config_path.resolve().parent
    log_level = str((raw.get("logging") or {}).get("level", "INFO")).upper()
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ConfigError(f"logging.level is not a valid level: {log_level!r}")

    config = AppConfig(
        paths=_build_paths(raw.get("paths"), base_dir),
        chunking=_build_section(ChunkingConfig, raw.get("chunking"), "chunking"),
        embedding=_build_section(EmbeddingConfig, raw.get("embedding"), "embedding"),
        retrieval=_build_section(RetrievalConfig, raw.get("retrieval"), "retrieval"),
        generation=_build_section(GenerationConfig, raw.get("generation"), "generation"),
        quiz=_build_section(QuizConfig, raw.get("quiz"), "quiz"),
        log_level=log_level,
    )
    logger.debug("Loaded config from %s", config_path)
    return config


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once, with a compact readable format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Third-party libraries are chatty at INFO; keep their noise down.
    # pypdf logs a WARNING for every minor PDF quirk; real failures still raise.
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    for noisy in ("httpx", "urllib3", "sentence_transformers", "transformers", "faiss"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_api_key(provider: str) -> str | None:
    """Return the API key for ``provider`` from the environment, or None.

    Loads ``<project>/.env`` first if it exists. Real environment variables
    (such as Hugging Face Space secrets) take precedence over the file.
    """
    if provider not in API_KEY_ENV_VARS:
        raise ConfigError(f"Unknown provider: {provider!r}")
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    for var in API_KEY_ENV_VARS[provider]:
        value = os.environ.get(var, "").strip()
        if value:
            return value
    return None
