"""Tests for deployment-related behaviour: env overrides, bring-your-own-key, CLI errors."""

from __future__ import annotations

import pytest
from fakes import FakeLLM, HashingEncoder

import notemind.llm_client as llm_module
import notemind.pipeline as pipeline_module
from notemind.config import ConfigError, load_config, run_cli
from notemind.llm_client import LLMConfigError
from notemind.pipeline import RAGPipeline


def test_env_overrides_choose_provider_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NOTEMIND_PROVIDER", "mistral")
    monkeypatch.setenv("NOTEMIND_MODEL", "mistral-large-latest")
    generation = load_config().generation
    assert generation.provider == "mistral"
    assert generation.model == "mistral-large-latest"


def test_invalid_env_provider_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NOTEMIND_PROVIDER", "chatgpt")
    with pytest.raises(ConfigError, match="generation.provider"):
        load_config()


def test_user_supplied_key_is_passed_to_the_client(app_config, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str | None] = []

    def fake_factory(config, api_key=None):
        seen.append(api_key)
        return FakeLLM(["ok"])

    monkeypatch.setattr(pipeline_module, "create_llm_client", fake_factory)
    pipe = RAGPipeline(app_config, embedder=HashingEncoder())
    pipe.set_api_key("  sk-user-key  ")
    _ = pipe.llm
    pipe.set_api_key("sk-user-key")  # unchanged: client is reused
    _ = pipe.llm
    pipe.set_api_key("")  # cleared: falls back to the environment
    _ = pipe.llm
    assert seen == ["sk-user-key", None]


def test_missing_sdk_gives_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str):
        raise ModuleNotFoundError(f"No module named '{name}'", name=name)

    monkeypatch.setattr(llm_module.importlib, "import_module", missing)
    with pytest.raises(LLMConfigError, match="pip install -r requirements.txt"):
        llm_module._import_sdk("openai", "openai")


def test_run_cli_prints_one_line_errors(capsys: pytest.CaptureFixture[str]) -> None:
    def bad_config() -> int:
        raise ConfigError("Duplicate key 'provider' in config.yaml")

    def missing_package() -> int:
        raise ModuleNotFoundError("No module named 'openai'", name="openai")

    assert run_cli(bad_config) == 1
    assert run_cli(missing_package) == 1
    assert run_cli(lambda: 0) == 0
    out = capsys.readouterr().out
    assert "Config error: Duplicate key 'provider'" in out
    assert "Missing package 'openai'" in out
    assert "Traceback" not in out


def test_deployment_requirements_match_local_pins() -> None:
    """app/requirements.txt (Streamlit Cloud) must pin the same versions as requirements.txt."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]

    def pins(path: Path) -> dict[str, str]:
        result = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if "==" in line and not line.startswith("-"):
                name, version = line.split("==", 1)
                result[name.strip().lower()] = version.strip()
        return result

    assert pins(root / "app" / "requirements.txt") == pins(root / "requirements.txt")
