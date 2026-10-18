"""Tests for the evaluation metrics in eval/run_eval.py (a bug here would corrupt every result)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "run_eval", Path(__file__).resolve().parents[1] / "eval" / "run_eval.py"
)
run_eval = importlib.util.module_from_spec(_SPEC)
sys.modules["run_eval"] = run_eval  # @dataclass looks the module up while it loads
_SPEC.loader.exec_module(run_eval)


def test_first_relevant_rank() -> None:
    retrieved = [("a.pdf", 1), ("b.pdf", 2), ("a.pdf", 3)]
    assert run_eval.first_relevant_rank(retrieved, {("a.pdf", 3)}) == 3
    assert run_eval.first_relevant_rank(retrieved, {("b.pdf", 2), ("a.pdf", 3)}) == 2
    assert run_eval.first_relevant_rank(retrieved, {("c.pdf", 1)}) is None


def test_recall_and_mrr_by_hand() -> None:
    ranks = [1, 3, None, 2]
    assert run_eval.recall_at_k(ranks, 1) == pytest.approx(0.25)
    assert run_eval.recall_at_k(ranks, 3) == pytest.approx(0.75)
    assert run_eval.mrr(ranks) == pytest.approx((1 + 1 / 3 + 0 + 1 / 2) / 4)
    assert run_eval.recall_at_k([], 5) == 0.0 and run_eval.mrr([]) == 0.0


def test_bootstrap_interval_contains_the_mean() -> None:
    values = [1.0] * 30 + [0.0] * 10
    low, high = run_eval.bootstrap_ci(values)
    assert low <= 0.75 <= high
    assert run_eval.bootstrap_ci([1.0, 1.0]) == (1.0, 1.0)


def test_threshold_sweep_counts_refusals() -> None:
    questions = [
        run_eval.EvalQuestion("a1", "q", True, {("f.pdf", 1)}),
        run_eval.EvalQuestion("a2", "q", True, {("f.pdf", 2)}),
        run_eval.EvalQuestion("u1", "q", False),
    ]
    sims = {"a1": 0.6, "a2": 0.2, "u1": 0.1}
    rows = {r["threshold"]: r for r in run_eval.threshold_sweep(questions, sims, [0.15, 0.3])}
    assert rows[0.15] == {"threshold": 0.15, "correct_refusal": 1.0, "false_refusal": 0.0}
    assert rows[0.3] == {"threshold": 0.3, "correct_refusal": 1.0, "false_refusal": 0.5}


def test_eval_set_is_valid_and_rejects_bad_labels(tmp_path: Path) -> None:
    questions, pdf_dir = run_eval.load_eval_set(Path(__file__).resolve().parents[1] / "eval" / "eval_set.json")
    assert len(questions) >= 40 and pdf_dir.name == "demo"
    assert {q.answerable for q in questions} == {True, False}

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"questions": [{"id": "x", "question": "q", "answerable": True, "expected": []}]}))
    with pytest.raises(ValueError, match="expected page"):
        run_eval.load_eval_set(bad)


def test_stratified_sample_keeps_both_kinds() -> None:
    questions = [run_eval.EvalQuestion(f"a{i}", "q", True, {("f.pdf", 1)}) for i in range(40)]
    questions += [run_eval.EvalQuestion(f"u{i}", "q", False) for i in range(12)]
    sample = run_eval.stratified_sample(questions, 18)
    assert len(sample) == 18
    assert sum(not q.answerable for q in sample) >= 2
    assert len({q.id for q in sample}) == 18
    assert run_eval.stratified_sample(questions, 100) == questions


def test_llm_eval_stops_when_quota_is_exhausted() -> None:
    from notemind.llm_client import LLMRateLimitError
    from notemind.models import Answer

    class QuotaPipeline:
        calls = 0

        def ask(self, question: str) -> Answer:
            QuotaPipeline.calls += 1
            if QuotaPipeline.calls > 2:
                raise LLMRateLimitError("quota", retry_after=5580)
            return Answer(text="Fact [f.pdf, p.1].")

    questions = [run_eval.EvalQuestion(f"a{i}", "q", True, {("f.pdf", 1)}) for i in range(10)]
    result = run_eval.evaluate_with_llm(QuotaPipeline(), questions, delay=0)
    assert QuotaPipeline.calls == 3  # stopped instead of failing 7 more times
    assert result["n_ok"] == 2 and result["n_planned"] == 10
    assert "1 h 33 min" in result["stopped_reason"]
    assert result["answers_citing_expected_page"] == 1.0


def test_recommend_threshold_prefers_zero_false_refusals() -> None:
    # The user's real sweep (Oct 2026): 0.25 is the highest threshold with no false refusals.
    sweep = [
        {"threshold": 0.20, "correct_refusal": 0.083, "false_refusal": 0.0},
        {"threshold": 0.25, "correct_refusal": 0.25, "false_refusal": 0.0},
        {"threshold": 0.30, "correct_refusal": 0.333, "false_refusal": 0.025},
        {"threshold": 0.45, "correct_refusal": 0.833, "false_refusal": 0.10},
    ]
    assert run_eval.recommend_threshold(sweep) == 0.25
    assert run_eval.recommend_threshold(sweep, max_false_refusal=0.05) == 0.30
    assert run_eval.recommend_threshold([{"threshold": 0.1, "correct_refusal": 0, "false_refusal": 0.5}]) is None


def test_llm_eval_stops_after_repeated_failures() -> None:
    from notemind.llm_client import LLMRateLimitError
    from notemind.models import Answer

    class FlakyPipeline:
        calls = 0

        def ask(self, question: str) -> Answer:
            FlakyPipeline.calls += 1
            if FlakyPipeline.calls == 2:  # a single failure is tolerated...
                raise LLMRateLimitError("busy")
            if FlakyPipeline.calls >= 4:  # ...but three in a row stops the run
                raise LLMRateLimitError("upstream pool overloaded")
            return Answer(text="Fact [f.pdf, p.1].")

    questions = [run_eval.EvalQuestion(f"a{i}", "q", True, {("f.pdf", 1)}) for i in range(20)]
    result = run_eval.evaluate_with_llm(FlakyPipeline(), questions, delay=0)
    assert FlakyPipeline.calls == 6
    assert result["n_ok"] == 2 and result["n_errors"] == 4
    assert "3 calls in a row failed" in result["stopped_reason"]
