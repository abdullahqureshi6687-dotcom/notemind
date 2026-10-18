"""Tests for RAGPipeline, Generator, QuizGenerator and LLM retry logic.

All LLM calls go to FakeLLM and all embeddings to HashingEncoder, so these
run offline without an API key or a model download.
"""

from __future__ import annotations

import json
from unittest import mock

import pytest
from fakes import FakeLLM, HashingEncoder

import notemind.llm_client as llm_module
from notemind.config import AppConfig, GenerationConfig
from notemind.generator import Generator, extract_citations, is_refusal
from notemind.llm_client import (
    LLMAuthError,
    LLMRateLimitError,
    LLMTruncatedError,
    LLMUnavailableError,
)
from notemind.models import Page, SearchResult
from notemind.pipeline import RAGPipeline
from notemind.prompts import REFUSAL_TEXT, format_context
from notemind.quiz import QuizGenerationError, QuizGenerator, parse_quiz

ANSWER = "The learning rate controls the size of each step [ml.pdf, p.1]."


def quiz_json(n: int = 3, page: int = 1) -> str:
    return json.dumps(
        {
            "questions": [
                {
                    "question": f"Question {i} about gradient descent?",
                    "options": [f"Correct {i}", f"Wrong {i}a", f"Wrong {i}b", f"Wrong {i}c"],
                    "answer_index": 0,
                    "explanation": f"Explained on page {page}.",
                    "source_page": page,
                }
                for i in range(n)
            ]
        }
    )


@pytest.fixture
def pipeline(app_config: AppConfig, notes_pdf_bytes: bytes, encoder: HashingEncoder) -> RAGPipeline:
    pipe = RAGPipeline(app_config, embedder=encoder, llm=FakeLLM([ANSWER]))
    pipe.add_pdf_bytes(notes_pdf_bytes, "ml.pdf")
    pipe.build_index()
    return pipe


# ----------------------------------------------------------------- pipeline


def test_ask_returns_a_cited_answer(pipeline: RAGPipeline) -> None:
    answer = pipeline.ask("What does the learning rate control in gradient descent?")
    assert not answer.refused
    assert answer.text == ANSWER
    assert answer.invalid_citations == []
    assert answer.sources and answer.sources[0].chunk.citation == "[ml.pdf, p.1]"
    assert answer.max_similarity is not None


def test_low_similarity_refuses_without_calling_the_llm(pipeline: RAGPipeline) -> None:
    llm = pipeline.llm
    answer = pipeline.ask("photosynthesis in the chloroplasts of plants")
    assert answer.refused and answer.refusal_reason == "low_similarity"
    assert answer.text == REFUSAL_TEXT
    assert llm.calls == []  # type: ignore[attr-defined]


def test_ask_without_documents_or_question(app_config: AppConfig, encoder: HashingEncoder) -> None:
    pipe = RAGPipeline(app_config, embedder=encoder, llm=FakeLLM())
    assert pipe.ask("anything").refusal_reason == "no_index"
    assert pipe.ask("   ").refusal_reason == "empty_question"
    assert pipe.build_index() is None


def test_index_is_cached_across_sessions(
    app_config: AppConfig, notes_pdf_bytes: bytes, encoder: HashingEncoder
) -> None:
    first = RAGPipeline(app_config, embedder=encoder)
    first.add_pdf_bytes(notes_pdf_bytes, "ml.pdf")
    assert first.build_index().from_cache is False
    assert encoder.calls == 1

    second = RAGPipeline(app_config, embedder=encoder)
    second.add_pdf_bytes(notes_pdf_bytes, "ml.pdf")
    stats = second.build_index()
    assert stats.from_cache is True
    assert encoder.calls == 1  # nothing re-embedded


def test_config_changes_invalidate_only_when_needed(pipeline: RAGPipeline) -> None:
    pipeline.update_config(pipeline.config.with_overrides(retrieval={"mode": "bm25", "top_k": 2}))
    assert pipeline.is_indexed  # retrieval-only change keeps the index

    old_key = pipeline.index_key()
    pipeline.update_config(pipeline.config.with_overrides(chunking={"chunk_size": 300, "chunk_overlap": 50}))
    assert not pipeline.is_indexed and pipeline.needs_indexing
    assert pipeline.index_key() != old_key


def test_documents_add_replace_and_remove(pipeline: RAGPipeline, make_pdf) -> None:
    pipeline.add_pdf_bytes(make_pdf(["Different content about heaps and trees."]), "ml.pdf")
    assert len(pipeline.documents) == 1 and pipeline.documents[0].num_pages == 1
    assert not pipeline.is_indexed  # replaced file needs reindexing

    pipeline.remove_document("ml.pdf")
    assert pipeline.documents == [] and not pipeline.needs_indexing


def test_re_adding_identical_file_keeps_the_index(pipeline: RAGPipeline, notes_pdf_bytes: bytes) -> None:
    pipeline.add_pdf_bytes(notes_pdf_bytes, "ml.pdf")
    assert pipeline.is_indexed


def test_make_quiz_uses_selected_pages(pipeline: RAGPipeline) -> None:
    pipeline._llm = FakeLLM([quiz_json(2, page=2)])
    quiz = pipeline.make_quiz("ml.pdf", num_questions=2, start_page=2, end_page=2, seed=0)
    assert quiz.pages == [2]
    assert len(quiz.questions) == 2
    assert "--- Page 2 ---" in pipeline._llm.calls[0]["user"]
    assert "--- Page 1 ---" not in pipeline._llm.calls[0]["user"]
    with pytest.raises(KeyError):
        pipeline.make_quiz("missing.pdf")


# ----------------------------------------------------------------- generator


def test_citation_extraction_and_refusal_detection() -> None:
    text = "A [ml.pdf, p.3]. B [ML.pdf, p. 7]. C [ml.pdf, page 3]."
    assert extract_citations(text) == [("ml.pdf", 3), ("ML.pdf", 7)]
    assert is_refusal(REFUSAL_TEXT)
    assert is_refusal("I couldn’t find this in your notes")
    assert not is_refusal("Gradient descent steps downhill [ml.pdf, p.1]. " * 5 + "couldn't find this in your notes")


def test_generator_flags_invented_citations(chunks) -> None:
    results = [SearchResult(chunks[0], 0.8, 1, "faiss")]
    reply = f"Real fact {chunks[0].citation}. Invented fact [ml.pdf, p.99]."
    answer = Generator(FakeLLM([reply]), GenerationConfig()).answer("question?", results)
    assert answer.invalid_citations == ["[ml.pdf, p.99]"]


def test_generator_detects_model_refusal_and_handles_no_results(chunks) -> None:
    results = [SearchResult(chunks[0], 0.8, 1, "faiss")]
    answer = Generator(FakeLLM([REFUSAL_TEXT]), GenerationConfig()).answer("question?", results)
    assert answer.refused and answer.refusal_reason == "llm"
    assert Generator(FakeLLM(), GenerationConfig()).answer("question?", []).refusal_reason == "no_results"


def test_context_never_exceeds_the_character_cap(chunks) -> None:
    results = [SearchResult(c, 0.5, i + 1, "faiss") for i, c in enumerate(chunks)]
    for cap in (1000, 1500, 3000):
        context, used = format_context(results, cap)
        assert len(context) <= cap
        assert used == results[: len(used)]  # highest-ranked chunks are kept


# ----------------------------------------------------------------- quiz


PAGES = [Page("ml.pdf", i, f"Page {i} explains gradient descent and learning rates. " * 10) for i in (1, 2, 3)]


def test_quiz_parses_fenced_json_and_keeps_correct_answers_after_shuffle() -> None:
    llm = FakeLLM(["Here you go:\n```json\n" + quiz_json(3) + "\n```"])
    quiz = QuizGenerator(llm, seed=42).generate(PAGES, 3)
    assert len(quiz.questions) == 3
    assert all(q.correct_option.startswith("Correct") for q in quiz.questions)
    assert llm.calls[0]["json_mode"] is True


def test_quiz_repairs_invalid_json_once() -> None:
    bad = json.loads(quiz_json(3))
    bad["questions"][0]["options"] = ["Same", "Same", "Other", "Another"]
    llm = FakeLLM([json.dumps(bad), quiz_json(3)])
    quiz = QuizGenerator(llm).generate(PAGES, 3)
    assert len(quiz.questions) == 3
    assert "options must all be different" in llm.calls[1]["user"]


def test_quiz_fails_cleanly_after_two_invalid_replies() -> None:
    with pytest.raises(QuizGenerationError):
        QuizGenerator(FakeLLM(["not json", "still not json"])).generate(PAGES, 3)


def test_truncated_quiz_is_retried_with_a_bigger_budget() -> None:
    llm = FakeLLM([LLMTruncatedError("cut off", '{"questions": ['), quiz_json(3)])
    quiz = QuizGenerator(llm).generate(PAGES, 3)
    assert len(quiz.questions) == 3
    assert llm.calls[1]["max_tokens"] == 2 * llm.calls[0]["max_tokens"]


def test_quiz_validation_rules() -> None:
    with pytest.raises(ValueError):
        parse_quiz(json.dumps({"questions": [{"question": "Valid question?", "options": ["a", "b"],
                                              "answer_index": 0, "explanation": "Because."}]}))
    quiz = parse_quiz(json.dumps({"questions": [{"question": "Valid\nquestion?", "options": ["a\tx", "b", "c", "d"],
                                                 "answer_index": 3, "explanation": "Because."}]}))
    assert quiz.questions[0].question == "Valid question?"
    assert quiz.questions[0].options[0] == "a x"


def test_quiz_with_no_text_raises() -> None:
    with pytest.raises(QuizGenerationError):
        QuizGenerator(FakeLLM()).generate([Page("x.pdf", 1, "")], 3)


# ----------------------------------------------------------------- LLM retries


def test_retryable_errors_are_retried_then_succeed() -> None:
    llm = FakeLLM([LLMRateLimitError("429", retry_after=0.01), LLMUnavailableError("503"), "ok"],
                  config=GenerationConfig(max_retries=3))
    with mock.patch.object(llm_module.time, "sleep") as sleep:
        assert llm.generate("system", "user") == "ok"
    assert len(llm.calls) == 3
    assert sleep.call_count == 2


def test_non_retryable_errors_fail_immediately() -> None:
    llm = FakeLLM([LLMAuthError("bad key"), "never returned"], config=GenerationConfig(max_retries=3))
    with mock.patch.object(llm_module.time, "sleep") as sleep, pytest.raises(LLMAuthError):
        llm.generate("system", "user")
    assert len(llm.calls) == 1 and sleep.call_count == 0


def test_gives_up_after_max_retries() -> None:
    llm = FakeLLM([LLMUnavailableError("down")] * 3, config=GenerationConfig(max_retries=2))
    with mock.patch.object(llm_module.time, "sleep"), pytest.raises(LLMUnavailableError):
        llm.generate("system", "user")
    assert len(llm.calls) == 3


def test_missing_api_key_gives_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(llm_module, "get_api_key", lambda provider: None)
    with pytest.raises(llm_module.LLMConfigError, match="GEMINI_API_KEY"):
        llm_module.create_llm_client(GenerationConfig(provider="gemini"))


# ----------------------------------------------------------------- quotas


def test_exhausted_quota_is_not_retried() -> None:
    llm = FakeLLM([LLMRateLimitError("quota", retry_after=5580), "never returned"],
                  config=GenerationConfig(max_retries=3))
    with mock.patch.object(llm_module.time, "sleep") as sleep, pytest.raises(LLMRateLimitError) as info:
        llm.generate("system", "user")
    assert info.value.quota_exhausted
    assert len(llm.calls) == 1 and sleep.call_count == 0


def test_gemini_retry_delay_is_parsed_from_the_error() -> None:
    from google.genai import errors

    body = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                      "message": "Quota exceeded for metric: free_tier_requests, limit: 20. Please retry in 1h33m0.09s.",
                      "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "5580s"}]}}
    assert llm_module._gemini_retry_after(errors.ClientError(429, body)) == 5580
    body["error"]["details"] = []  # fall back to the "retry in ..." text
    assert llm_module._gemini_retry_after(errors.ClientError(429, body)) == pytest.approx(5580.09)


def test_format_wait() -> None:
    assert llm_module.format_wait(5580) == "1 h 33 min"
    assert llm_module.format_wait(150) == "2 min"
    assert llm_module.format_wait(42) == "42 s"


# ----------------------------------------------------------------- config file


def test_duplicate_config_keys_are_rejected(tmp_path) -> None:
    from notemind.config import ConfigError, load_config

    bad = tmp_path / "config.yaml"
    bad.write_text("generation:\n  provider: openrouter\n\n  provider: gemini\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"Duplicate key .provider. in config.yaml \(line 4"):
        load_config(bad)
