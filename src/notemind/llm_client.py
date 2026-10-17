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

import logging
import random
import time
from abc import ABC, abstractmethod

from notemind.config import API_KEY_ENV_VARS, GenerationConfig, get_api_key

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


class LLMUnavailableError(LLMError):
    """Timeout, connection problem or provider-side error (5xx)."""

    retryable = True


class LLMEmptyResponseError(LLMError):
    """The model returned no text (blocked, or ran out of tokens)."""


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
                if not exc.retryable or attempt == attempts:
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
        import anthropic

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
            logger.warning("Answer was cut off at max_tokens=%d", max_tokens)
        return text


class GroqClient(_StainlessErrorMixin, LLMClient):
    """Open models hosted by Groq via the ``groq`` SDK (OpenAI-style API)."""

    provider = "groq"

    def __init__(self, model: str, config: GenerationConfig, api_key: str) -> None:
        super().__init__(model, config)
        import groq

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
        from google import genai
        from google.genai import errors, types

        self._types = types
        self._errors = errors
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
        try:
            response = self._client.models.generate_content(
                model=self.model, contents=user, config=gen_config
            )
        except self._errors.APIError as exc:
            raise _classify_status(exc.code, str(exc.message or exc), _retry_after_seconds(exc)) from exc
        except httpx.TransportError as exc:  # timeouts, DNS, connection resets
            raise LLMUnavailableError(f"Network problem or timeout: {exc}") from exc

        text = (response.text or "").strip()
        if not text:
            candidate = response.candidates[0] if response.candidates else None
            reason = getattr(candidate, "finish_reason", None) if candidate else "no candidates"
            raise LLMEmptyResponseError(
                f"Gemini returned no text (finish_reason={reason}). 'MAX_TOKENS' means the "
                "model used its budget thinking: increase generation.max_tokens. 'SAFETY' "
                "means the request was blocked."
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
    cls = _PROVIDERS.get(config.provider)
    if cls is None:
        raise LLMConfigError(f"Unknown provider {config.provider!r}")
    key = api_key or get_api_key(config.provider)
    if not key:
        env_vars = " or ".join(API_KEY_ENV_VARS[config.provider])
        raise LLMConfigError(
            f"No API key for {config.provider}. Set {env_vars} in your .env file "
            "(or as a Secret on Hugging Face Spaces)."
        )
    return cls(config.model, config, key)  # type: ignore[call-arg]


def _main() -> int:
    """Developer CLI: check that the configured provider, model and key work.

    Usage: python -m notemind.llm_client
    """
    from notemind.config import load_config, setup_logging

    config = load_config()
    setup_logging(config.log_level)
    gen = config.generation
    print(f"Provider: {gen.provider}   Model: {gen.model}")
    try:
        client = create_llm_client(gen)
        started = time.perf_counter()
        reply = client.generate("You are a connection test.", "Reply with exactly: OK", max_tokens=200)
    except LLMError as exc:
        print(f"FAILED ({type(exc).__name__}): {exc}")
        return 1
    print(f"Reply: {reply!r}   ({time.perf_counter() - started:.1f}s)")
    print("Connection works.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
