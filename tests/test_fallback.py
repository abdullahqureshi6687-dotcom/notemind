"""Tests for backup models: when the main model fails, the next one is tried."""

from __future__ import annotations

from collections.abc import Callable
from unittest import mock

import pytest

import notemind.llm_client as llm_module
from notemind.config import ConfigError, GenerationConfig, load_config
from notemind.llm_client import (
    LLMAuthError,
    LLMBadRequestError,
    LLMClient,
    LLMRateLimitError,
    LLMUnavailableError,
)


def make_config(max_retries: int = 2, backups: list[str] | None = None) -> GenerationConfig:
    return GenerationConfig(
        provider="openrouter",
        models={"openrouter": "main"},
        fallback_models={"openrouter": ["backup-1", "backup-2"] if backups is None else backups},
        max_retries=max_retries,
    )


class PerModelLLM(LLMClient):
    """Fake client whose behaviour depends on which model is being called."""

    provider = "openrouter"

    def __init__(self, behaviour: dict[str, Callable[[], str]], config: GenerationConfig) -> None:
        super().__init__(config.model, config)
        self.behaviour = behaviour
        self.called: list[str] = []

    def _call(self, system: str, user: str, *, json_mode: bool, max_tokens: int) -> str:
        self.called.append(self.model)
        return self.behaviour[self.model]()


def fails(exc: Exception) -> Callable[[], str]:
    def _raise() -> str:
        raise exc
    return _raise


def test_model_chain_puts_main_model_first_without_duplicates() -> None:
    config = make_config(backups=["backup-1", "main", " backup-1 ", "backup-2"])
    assert config.model_chain == ["main", "backup-1", "backup-2"]
    assert GenerationConfig(provider="openrouter", models={"openrouter": "m"}).model_chain == ["m"]


def test_invalid_fallback_models_are_rejected() -> None:
    with pytest.raises(ConfigError, match="fallback_models"):
        GenerationConfig(provider="openrouter", models={"openrouter": "m"},
                         fallback_models={"openrouter": "not-a-list"})  # type: ignore[dict-item]


def test_retired_model_falls_back_and_backup_becomes_current() -> None:
    llm = PerModelLLM(
        {"main": fails(LLMBadRequestError("404 Function not found")), "backup-1": lambda: "answer"},
        make_config(),
    )
    with mock.patch.object(llm_module.time, "sleep") as sleep:
        assert llm.generate("system", "user") == "answer"
        assert llm.called == ["main", "backup-1"]  # main tried once, no retries
        assert sleep.call_count == 0
        assert llm.model == "backup-1"

        llm.generate("system", "user")  # later calls start with the working backup
    assert llm.called == ["main", "backup-1", "backup-1"]


def test_rate_limit_and_outage_also_fall_back() -> None:
    llm = PerModelLLM(
        {
            "main": fails(LLMRateLimitError("429")),
            "backup-1": fails(LLMUnavailableError("503")),
            "backup-2": lambda: "ok",
        },
        make_config(),
    )
    with mock.patch.object(llm_module.time, "sleep"):
        assert llm.generate("s", "u") == "ok"
    assert llm.called == ["main", "backup-1", "backup-2"]


def test_key_errors_are_not_retried_on_other_models() -> None:
    llm = PerModelLLM({"main": fails(LLMAuthError("401"))}, make_config())
    with pytest.raises(LLMAuthError):
        llm.generate("s", "u")
    assert llm.called == ["main"]
    assert llm.model == "main"


def test_last_model_gets_full_retries_then_raises() -> None:
    llm = PerModelLLM(
        {
            "main": fails(LLMBadRequestError("404")),
            "backup-1": fails(LLMBadRequestError("404")),
            "backup-2": fails(LLMUnavailableError("503")),
        },
        make_config(max_retries=2),
    )
    with mock.patch.object(llm_module.time, "sleep"), pytest.raises(LLMUnavailableError):
        llm.generate("s", "u")
    assert llm.called == ["main", "backup-1", "backup-2", "backup-2", "backup-2"]
    assert llm.model == "main"  # nothing worked, so the main model stays current


def test_without_backups_retry_policy_is_unchanged() -> None:
    llm = PerModelLLM({"main": fails(LLMUnavailableError("503"))}, make_config(max_retries=2, backups=[]))
    with mock.patch.object(llm_module.time, "sleep"), pytest.raises(LLMUnavailableError):
        llm.generate("s", "u")
    assert llm.called == ["main"] * 3


def test_env_sets_fallback_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NOTEMIND_PROVIDER", "openrouter")
    monkeypatch.setenv("NOTEMIND_MODEL", "main/model:free")
    monkeypatch.setenv("NOTEMIND_FALLBACK_MODELS", " backup/a:free , openrouter/free ,")
    generation = load_config().generation
    assert generation.model_chain == ["main/model:free", "backup/a:free", "openrouter/free"]


def test_empty_env_fallbacks_turn_backups_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NOTEMIND_PROVIDER", "openrouter")
    monkeypatch.setenv("NOTEMIND_FALLBACK_MODELS", "")
    generation = load_config().generation
    assert generation.model_chain == [generation.model]
