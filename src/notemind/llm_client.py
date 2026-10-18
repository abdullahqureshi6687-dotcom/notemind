"""Provider-agnostic LLM client with retries and clean error handling.

The rest of the code only ever calls ``client.generate(system, user)``. Each
provider (Anthropic, Gemini, Groq) is a small subclass that translates that
call into its own SDK and translates its SDK's exceptions into ours. This is
the Strategy pattern: switching providers is a one-line config change and the
Generator / QuizGenerator never import a provider SDK.

Retries:
    Rate limits (HTTP 429), server errors (5xx, Anthropic's 529 "overloaded"),
    timeouts and connection errors are retried with exponential backoff plus
    jitter, honouring a ``Retry-After`` header when the provider sends one.
    Bad API keys, unknown model names and malformed requests are NOT retried:
    retrying can't fix them, so the user gets a clear message immediately.
    The SDKs' own built-in retries are disabled (``max_retries=0``) so the
    retry policy lives in exactly one place.
"""

from __future__ import annotations

import importlib
import logging
import random
import re
import time
from abc import ABC, abstractmethod
from types import ModuleType

from notemind.config import (
    API_KEY_ENV_VARS,
    OPENAI_COMPATIBLE_BASE_URLS,
    GenerationConfig,
    get_api_key,
    provider_needs_key,
)

logger = logging.getLogger(__name__)

MAX_BACKOFF_SECONDS: float = 30.0
BASE_BACKOFF_SECONDS: float = 1.0


class LLMError(Exception):
    """Base class for LLM failures. ``retryable`` says whether to try again."""

    retryable: bool = False

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class LLMConfigError(LLMError):
    """Missing API key or unknown provider. Fix the setup, then retry."""


class LLMAuthError(LLMError):
    """The API key was rejected (HTTP 401/403)."""


class LLMBadRequestError(LLMError):
    """The request itself is wrong, e.g. an unknown model name (HTTP 400/404)."""


class LLMRateLimitError(LLMError):
    """Too many requests or quota exceeded (HTTP 429)."""

    retryable = True

    @property
    def quota_exhausted(self) -> bool:
        """True when the provider asks us to wait longer than we would ever back off.

        That means a quota (for example a daily free-tier limit) is used up,
        not a short burst limit, so retrying now is pointless.
        """
        return self.retry_after is not None and self.retry_after > MAX_BACKOFF_SECONDS


class LLMUnavailableError(LLMError):
    """Timeout, connection problem or provider-side error (5xx)."""

    retryable = True


class LLMEmptyResponseError(LLMError):
    """The model returned no text (blocked, or ran out of tokens)."""


class LLMTruncatedError(LLMError):
    """JSON output was cut off at the token limit, so it can't be parsed.

    Not retried automatically: the same budget would fail the same way. The
    caller (QuizGenerator) retries with a larger ``max_tokens`` instead.
    """

    def __init__(self, message: str, partial_text: str = "") -> None:
        super().__init__(message)
        self.partial_text = partial_text


def _import_sdk(module: str, package: str) -> ModuleType:
    """Import a provider SDK, turning a missing package into a clear LLMConfigError."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise LLMConfigError(
            f"The '{package}' package is not installed. Run: pip install -r requirements.txt"
        ) from exc


def format_wait(seconds: float) -> str:
    """Human-readable duration, e.g. 5580 -> '1 h 33 min', 42 -> '42 s'."""
    seconds = max(0, int(round(seconds)))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours} h {minutes} min"
    if minutes:
        return f"{minutes} min"
    return f"{secs} s"


_DURATION = re.compile(r"(?:(\d+)h)?\s*(?:(\d+)m(?!s))?\s*(?:([\d.]+)s)?", re.IGNORECASE)
_RETRY_IN = re.compile(r"retry in\s+([\dhms.\s]+)", re.IGNORECASE)


def _parse_duration(text: str | None) -> float | None:
    """Parse '5580s', '1h33m0.09s' or '45.5s' into seconds."""
    if not text:
        return None
    match = _DURATION.fullmatch(text.strip())
    if not match or not any(match.groups()):
        return None
    hours, minutes, secs = match.groups()
    return int(hours or 0) * 3600 + int(minutes or 0) * 60 + float(secs or 0)


def _gemini_retry_after(exc: Exception) -> float | None:
    """Gemini puts the wait in a RetryInfo detail ('5580s') and in the message ('retry in 1h33m')."""
    details = getattr(exc, "details", None)
    if isinstance(details, dict):
        for item in (details.get("error") or {}).get("details") or []:
            if isinstance(item, dict) and str(item.get("@type", "")).endswith("RetryInfo"):
                parsed = _parse_duration(item.get("retryDelay"))
                if parsed is not None:
                    return parsed
    match = _RETRY_IN.search(str(getattr(exc, "message", "") or exc))
    return _parse_duration(match.group(1).rstrip(". ")) if match else None


def _retry_after_seconds(exc: Exception) -> float | None:
    """Read a Retry-After header from an SDK exception, if present."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("retry-after")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _classify_status(status: int | None, message: str, retry_after: float | None) -> LLMError:
    """Map an HTTP status code to one of our exception types."""
    if status == 402:
        return LLMAuthError(f"Payment required (402): this model or account has no free credits left. {message}")
    if status in (401, 403):
        return LLMAuthError(f"API key rejected ({status}). Check your .env file. {message}")
    if status == 429:
        return LLMRateLimitError(f"Rate limit or quota exceeded (429). {message}", retry_after)
    if status in (400, 404, 422):
        return LLMBadRequestError(
            f"Request rejected ({status}). Is the model name in config.yaml correct? {message}"
        )
    if status is not None and status >= 500:
        return LLMUnavailableError(f"Provider error ({status}). {message}", retry_after)
    return LLMError(f"Unexpected API error ({status}). {message}")


class LLMClient(ABC):
    """Common interface: ``generate(system, user) -> str`` with retries."""

    provider: str = ""

    def __init__(self, model: str, config: GenerationConfig) -> None:
        self.model = model
        self.config = config

    def generate(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        """Send one system + user message and return the model's text.

        Args:
            system: Instructions (the "rules").
            user: The request, including retrieved context.
            json_mode: Ask the provider for JSON output where supported. The
                caller must still validate the result.
            max_tokens: Override ``config.max_tokens`` for this call.

        Raises:
            LLMError: A subclass describing what went wrong, after retries.
        """
        max_tokens = max_tokens or self.config.max_tokens
        attempts = self.config.max_retries + 1
        for attempt in range(1, attempts + 1):
            started = time.perf_counter()
            try:
                text = self._call(system, user, json_mode=json_mode, max_tokens=max_tokens)
                logger.info(
                    "%s/%s answered in %.1fs (%d chars)",
                    self.provider,
                    self.model,
                    time.perf_counter() - started,
                    len(text),
                )
                return text
            except LLMError as exc:
                too_long = exc.retry_after is not None and exc.retry_after > MAX_BACKOFF_SECONDS
                if not exc.retryable or attempt == attempts or too_long:
                    logger.error("%s call failed (%s): %s", self.provider, type(exc).__name__, exc)
                    raise
                delay = self._backoff(attempt, exc.retry_after)
                logger.warning(
                    "%s call failed (%s), retry %d/%d in %.1fs",
                    self.provider,
                    type(exc).__name__,
                    attempt,
                    attempts - 1,
                    delay,
                )
                time.sleep(delay)
        raise LLMError("unreachable")  # pragma: no cover

    @staticmethod
    def _backoff(attempt: int, retry_after: float | None) -> float:
        """Exponential backoff with jitter: ~1s, 2s, 4s ... capped at 30s."""
        if retry_after is not None:
            return min(max(retry_after, 0.0), MAX_BACKOFF_SECONDS)
        delay = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
        return min(delay * random.uniform(0.5, 1.5), MAX_BACKOFF_SECONDS)

    @abstractmethod
    def _call(self, system: str, user: str, *, json_mode: bool, max_tokens: int) -> str:
        """One provider request. Must raise LLMError subclasses, never SDK errors."""


class _StainlessErrorMixin:
    """Anthropic and Groq SDKs share the same exception hierarchy."""

    @staticmethod
    def _translate(sdk, exc: Exception) -> LLMError:
        if isinstance(exc, (sdk.APITimeoutError, sdk.APIConnectionError)):
            return LLMUnavailableError(f"Network problem or timeout: {exc}")
        if isinstance(exc, sdk.APIStatusError):
            message = getattr(exc, "message", str(exc))
            return _classify_status(exc.status_code, message, _retry_after_seconds(exc))
        return LLMError(f"Unexpected error: {exc}")


class AnthropicClient(_StainlessErrorMixin, LLMClient):
    """Claude via the official ``anthropic`` SDK."""

    provider = "anthropic"

    def __init__(self, model: str, config: GenerationConfig, api_key: str) -> None:
        super().__init__(model, config)
        anthropic = _import_sdk("anthropic", "anthropic")

        self._sdk = anthropic
        self._client = anthropic.Anthropic(
            api_key=api_key, timeout=config.timeout_seconds, max_retries=0
        )

    def _call(self, system: str, user: str, *, json_mode: bool, max_tokens: int) -> str:
        # No temperature: the current Messages API no longer accepts it.
        # JSON is requested in the prompt and validated by the caller.
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except self._sdk.APIError as exc:
            raise self._translate(self._sdk, exc) from exc

        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        ).strip()
        if not text:
            raise LLMEmptyResponseError(
                f"Claude returned no text (stop_reason={response.stop_reason})."
            )
        if response.stop_reason == "max_tokens":
            if json_mode:
                raise LLMTruncatedError(f"Output cut off at max_tokens={max_tokens}", text)
            logger.warning("Answer was cut off at max_tokens=%d", max_tokens)
        return text


class GroqClient(_StainlessErrorMixin, LLMClient):
    """Open models hosted by Groq via the ``groq`` SDK (OpenAI-style API)."""

    provider = "groq"

    def __init__(self, model: str, config: GenerationConfig, api_key: str) -> None:
        super().__init__(model, config)
        groq = _import_sdk("groq", "groq")

        self._sdk = groq
        self._client = groq.Groq(api_key=api_key, timeout=config.timeout_seconds, max_retries=0)

    def _call(self, system: str, user: str, *, json_mode: bool, max_tokens: int) -> str:
        kwargs: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.config.temperature,
            "max_completion_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if "gpt-oss" in self.model:
            # Reasoning models spend completion tokens on hidden reasoning;
            # "low" keeps answers fast and leaves room for the actual text.
            kwargs["reasoning_effort"] = "low"
        try:
            response = self._client.chat.completions.create(**kwargs)
        except self._sdk.APIError as exc:
            raise self._translate(self._sdk, exc) from exc

        choice = response.choices[0] if response.choices else None
        text = (choice.message.content or "").strip() if choice else ""
        if text and choice.finish_reason == "length":
            if json_mode:
                raise LLMTruncatedError(f"Output cut off at max_tokens={max_tokens}", text)
            logger.warning("Answer was cut off at max_tokens=%d", max_tokens)
        if not text:
            reason = choice.finish_reason if choice else "no choices"
            raise LLMEmptyResponseError(
                f"Groq returned no text (finish_reason={reason}). "
                "If it is 'length', increase generation.max_tokens."
            )
        return text


class GeminiClient(LLMClient):
    """Google Gemini via the ``google-genai`` SDK."""

    provider = "gemini"

    def __init__(self, model: str, config: GenerationConfig, api_key: str) -> None:
        super().__init__(model, config)
        genai = _import_sdk("google.genai", "google-genai")
        errors = _import_sdk("google.genai.errors", "google-genai")
        types = _import_sdk("google.genai.types", "google-genai")

        self._types = types
        self._errors = errors
        # Sent with each request until the model rejects it (then disabled).
        self._thinking_level = config.gemini_thinking_level
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=int(config.timeout_seconds * 1000)),  # ms
        )

    def _call(self, system: str, user: str, *, json_mode: bool, max_tokens: int) -> str:
        import httpx

        types = self._types
        gen_config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.config.temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json" if json_mode else None,
        )
        if self._thinking_level:
            gen_config.thinking_config = types.ThinkingConfig(
                thinking_level=self._thinking_level.upper()
            )
        try:
            response = self._client.models.generate_content(
                model=self.model, contents=user, config=gen_config
            )
        except self._errors.APIError as exc:
            retry_after = _retry_after_seconds(exc) or _gemini_retry_after(exc)
            if exc.code == 400 and self._thinking_level and "think" in str(exc).lower():
                # This model doesn't accept a thinking level: stop sending it.
                logger.warning(
                    "%s rejected thinking_level=%s; continuing without it", self.model, self._thinking_level
                )
                self._thinking_level = None
                return self._call(system, user, json_mode=json_mode, max_tokens=max_tokens)
            raise _classify_status(exc.code, str(exc.message or exc), retry_after) from exc
        except httpx.TransportError as exc:  # timeouts, DNS, connection resets
            raise LLMUnavailableError(f"Network problem or timeout: {exc}") from exc

        text = (response.text or "").strip()
        candidate = response.candidates[0] if response.candidates else None
        reason = getattr(candidate, "finish_reason", None) if candidate else None
        truncated = str(getattr(reason, "value", reason)) == "MAX_TOKENS"
        usage = response.usage_metadata
        thoughts = getattr(usage, "thoughts_token_count", None) if usage else None
        output = getattr(usage, "candidates_token_count", None) if usage else None
        budget_note = (
            f" (budget {max_tokens}: {thoughts or 0} thinking tokens, {output or 0} output tokens)"
        )

        if not text:
            raise LLMEmptyResponseError(
                f"Gemini returned no text (finish_reason={reason}){budget_note}. 'MAX_TOKENS' "
                "means the budget ran out: increase generation.max_tokens or lower "
                "gemini_thinking_level. 'SAFETY' means the request was blocked."
            )
        if truncated:
            if json_mode:
                raise LLMTruncatedError(f"Output cut off at the token limit{budget_note}", text)
            logger.warning("Answer was cut off at the token limit%s", budget_note)
        return text


class OpenAICompatibleClient(_StainlessErrorMixin, LLMClient):
    """Any provider that implements the OpenAI chat-completions API.

    One class covers OpenRouter, Mistral and a local Ollama server: only the
    base URL, API key and model ID differ. This is why adding a provider is a
    config change rather than new code.
    """

    def __init__(self, model: str, config: GenerationConfig, api_key: str, provider: str,
                 base_url: str) -> None:
        super().__init__(model, config)
        openai = _import_sdk("openai", "openai")

        self.provider = provider
        self._sdk = openai
        self._json_mode_supported = True  # switched off if the endpoint rejects it
        headers = {"X-Title": "NoteMind"} if provider == "openrouter" else None
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=config.timeout_seconds,
            max_retries=0,
            default_headers=headers,
        )

    def _call(self, system: str, user: str, *, json_mode: bool, max_tokens: int) -> str:
        kwargs: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.config.temperature,
            "max_tokens": max_tokens,
        }
        use_json = json_mode and self._json_mode_supported
        if use_json:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = self._client.chat.completions.create(**kwargs)
        except self._sdk.BadRequestError as exc:
            if use_json and ("response_format" in str(exc) or "json" in str(exc).lower()):
                # Some models behind these APIs don't support JSON mode. The
                # quiz parser validates (and repairs) the JSON anyway.
                logger.warning("%s/%s rejected JSON mode; retrying without it", self.provider, self.model)
                self._json_mode_supported = False
                return self._call(system, user, json_mode=json_mode, max_tokens=max_tokens)
            raise self._translate(self._sdk, exc) from exc
        except self._sdk.APIError as exc:
            raise self._translate(self._sdk, exc) from exc

        choice = response.choices[0] if response.choices else None
        text = (choice.message.content or "").strip() if choice and choice.message else ""
        reason = getattr(choice, "finish_reason", None) if choice else "no choices"
        if text and reason == "length":
            if json_mode:
                raise LLMTruncatedError(f"Output cut off at max_tokens={max_tokens}", text)
            logger.warning("Answer was cut off at max_tokens=%d", max_tokens)
        if not text:
            raise LLMEmptyResponseError(
                f"{self.provider} returned no text (finish_reason={reason}). "
                "Free models are sometimes overloaded: try again or pick another model."
            )
        return text


_PROVIDERS: dict[str, type[LLMClient]] = {
    "anthropic": AnthropicClient,
    "gemini": GeminiClient,
    "groq": GroqClient,
}


def create_llm_client(config: GenerationConfig, api_key: str | None = None) -> LLMClient:
    """Build the client for ``config.provider``.

    Args:
        config: Generation settings (provider, model, retries, timeout).
        api_key: Explicit key; if omitted it is read from the environment.

    Raises:
        LLMConfigError: If the provider is unknown or no API key is available.
    """
    provider = config.provider
    if provider not in _PROVIDERS and provider not in OPENAI_COMPATIBLE_BASE_URLS:
        raise LLMConfigError(f"Unknown provider {provider!r}")

    key = api_key or get_api_key(provider)
    if not key and provider_needs_key(provider):
        env_vars = " or ".join(API_KEY_ENV_VARS[provider])
        raise LLMConfigError(
            f"No API key for {provider}. Set {env_vars} in your .env file "
            "(or as a Secret on Hugging Face Spaces)."
        )

    if provider in OPENAI_COMPATIBLE_BASE_URLS:
        # Local servers such as Ollama ignore the key, but the SDK requires one.
        return OpenAICompatibleClient(config.model, config, key or "not-needed", provider, config.base_url)
    return _PROVIDERS[provider](config.model, config, key)  # type: ignore[call-arg]


def _main() -> int:
    """Developer CLI: check that the configured provider, model and key work.

    Usage: python -m notemind.llm_client
    """
    from notemind.config import load_config, setup_logging

    config = load_config()
    setup_logging(config.log_level)
    gen = config.generation
    print(f"Provider: {gen.provider}   Model: {gen.model}" + (f"   Endpoint: {gen.base_url}" if gen.base_url else ""))
    try:
        client = create_llm_client(gen)
        started = time.perf_counter()
        reply = client.generate("You are a connection test.", "Reply with exactly: OK", max_tokens=200)
    except LLMError as exc:
        print(f"FAILED ({type(exc).__name__}): {exc}")
        if gen.provider == "ollama":
            print("Is Ollama running? Install it from ollama.com, then run: ollama pull " + gen.model)
        return 1
    print(f"Reply: {reply!r}   ({time.perf_counter() - started:.1f}s)")
    print("Connection works.")
    return 0


if __name__ == "__main__":
    from notemind.config import run_cli

    raise SystemExit(run_cli(_main))
