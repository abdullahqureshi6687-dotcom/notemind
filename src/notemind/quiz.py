"""Multiple-choice quiz generation with validated JSON output.

LLMs usually return the JSON you asked for, but "usually" is not good enough
for code that renders it. So the output goes through three layers:

1. **Extraction** - strip code fences and surrounding prose, keep the JSON object.
2. **Validation** - Pydantic models enforce the schema: exactly 4 distinct
   options, an answer index in range, non-empty text, a real page number.
3. **One repair attempt** - if validation fails, the exact error is sent back
   to the model once, asking it to fix its output. If that also fails, a clear
   :class:`QuizGenerationError` is raised instead of rendering garbage.

Options are then shuffled with a seeded RNG because LLMs tend to put the
correct answer in the same position; shuffling removes that pattern.
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections.abc import Sequence

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from notemind.config import QuizConfig
from notemind.llm_client import LLMClient
from notemind.models import Page
from notemind.prompts import (
    QUIZ_SYSTEM_PROMPT,
    build_quiz_prompt,
    build_quiz_repair_prompt,
    format_pages_for_quiz,
)

logger = logging.getLogger(__name__)

NUM_OPTIONS: int = 4


class QuizGenerationError(Exception):
    """Raised when a valid quiz could not be produced."""


class QuizQuestion(BaseModel):
    """One multiple-choice question."""

    question: str = Field(min_length=5)
    options: list[str] = Field(min_length=NUM_OPTIONS, max_length=NUM_OPTIONS)
    answer_index: int = Field(ge=0, lt=NUM_OPTIONS)
    explanation: str = Field(min_length=5)
    source_page: int | None = Field(default=None, ge=1)

    @field_validator("question", "explanation")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()

    @field_validator("options")
    @classmethod
    def _distinct_options(cls, options: list[str]) -> list[str]:
        cleaned = [str(opt).strip() for opt in options]
        if any(not opt for opt in cleaned):
            raise ValueError("options must not be empty")
        if len({opt.lower() for opt in cleaned}) != len(cleaned):
            raise ValueError("options must all be different")
        return cleaned

    @property
    def correct_option(self) -> str:
        return self.options[self.answer_index]


class Quiz(BaseModel):
    """A list of questions plus where they came from."""

    questions: list[QuizQuestion] = Field(min_length=1)
    source: str = ""
    pages: list[int] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _no_duplicate_questions(self) -> "Quiz":
        seen = [q.question.lower() for q in self.questions]
        if len(set(seen)) != len(seen):
            raise ValueError("questions must not repeat")
        return self


def extract_json_object(text: str) -> str:
    """Pull the JSON value out of a model response.

    Handles ```json fences and text before/after the JSON. Accepts an object
    ``{...}`` or a bare list ``[...]``, whichever starts first.

    Raises:
        ValueError: If no JSON object or list is present.
    """
    fenced = re.search(r"```(?:json)?\s*([\[{].*[\]}])\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    starts = [pos for pos in (text.find("{"), text.find("[")) if pos != -1]
    if not starts:
        raise ValueError("response does not contain a JSON object")
    start = min(starts)
    closing = "}" if text[start] == "{" else "]"
    end = text.rfind(closing)
    if end <= start:
        raise ValueError("response does not contain a complete JSON object")
    return text[start : end + 1]


def parse_quiz(text: str) -> Quiz:
    """Parse and validate raw model output into a :class:`Quiz`.

    Raises:
        ValueError: With a short, model-readable description of the problem.
    """
    try:
        data = json.loads(extract_json_object(text))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc.msg} at position {exc.pos}") from exc
    if isinstance(data, list):  # model returned the list without the wrapper
        data = {"questions": data}
    try:
        return Quiz.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()[:8]
        )
        raise ValueError(f"schema errors: {problems}") from exc


def filter_pages(
    pages: Sequence[Page],
    source: str,
    start_page: int | None = None,
    end_page: int | None = None,
) -> list[Page]:
    """Text pages of one PDF within an inclusive page range (None = open-ended)."""
    return [
        p
        for p in pages
        if p.source == source
        and not p.is_empty
        and (start_page is None or p.page_number >= start_page)
        and (end_page is None or p.page_number <= end_page)
    ]


class QuizGenerator:
    """Generate validated multiple-choice quizzes from note pages.

    Example::

        quiz_gen = QuizGenerator(llm, cfg.quiz)
        pages = filter_pages(all_pages, "lecture3.pdf", 2, 6)
        quiz = quiz_gen.generate(pages, num_questions=5)
    """

    def __init__(self, llm: LLMClient, config: QuizConfig | None = None, seed: int | None = None) -> None:
        self.llm = llm
        self.config = config or QuizConfig()
        self._rng = random.Random(seed)

    def generate(self, pages: Sequence[Page], num_questions: int | None = None) -> Quiz:
        """Create a quiz from the given pages.

        Args:
            pages: Pages to base questions on (usually one PDF or a page range).
            num_questions: How many questions; defaults to the config value and
                is clamped to ``[1, max_num_questions]``.

        Raises:
            QuizGenerationError: If there is no text, or the model fails to
                produce valid JSON after one repair attempt.
            LLMError: If the provider call itself fails.
        """
        n = num_questions or self.config.default_num_questions
        n = max(1, min(n, self.config.max_num_questions))

        text_pages = [p for p in pages if not p.is_empty]
        if not text_pages:
            raise QuizGenerationError("The selected pages contain no text to make a quiz from.")

        sources = sorted({p.source for p in text_pages})
        source_label = ", ".join(sources)
        notes, truncated = format_pages_for_quiz(text_pages, self.config.max_source_chars)
        warnings: list[str] = []
        if truncated:
            warnings.append(
                "The selection was too long, so only the first part was used. "
                "Choose a smaller page range to cover later pages."
            )

        prompt = build_quiz_prompt(notes, n, source_label)
        max_tokens = 400 + 300 * n  # JSON for one question is roughly 150-250 tokens
        raw = self.llm.generate(QUIZ_SYSTEM_PROMPT, prompt, json_mode=True, max_tokens=max_tokens)

        try:
            quiz = parse_quiz(raw)
        except ValueError as first_error:
            logger.warning("Quiz output invalid (%s); asking the model to repair it", first_error)
            repair = prompt + "\n\n" + build_quiz_repair_prompt(raw, str(first_error), n)
            raw = self.llm.generate(QUIZ_SYSTEM_PROMPT, repair, json_mode=True, max_tokens=max_tokens)
            try:
                quiz = parse_quiz(raw)
            except ValueError as second_error:
                raise QuizGenerationError(
                    f"The model did not return a valid quiz after a repair attempt ({second_error}). "
                    "Try again, or choose fewer questions."
                ) from second_error

        return self._finalize(quiz, text_pages, n, source_label, warnings)

    def _finalize(
        self,
        quiz: Quiz,
        pages: Sequence[Page],
        requested: int,
        source_label: str,
        warnings: list[str],
    ) -> Quiz:
        """Trim extras, fix page numbers, shuffle options, attach metadata."""
        valid_pages = {p.page_number for p in pages}
        questions = quiz.questions[:requested]
        if len(questions) < requested:
            warnings.append(f"The model produced {len(questions)} of {requested} requested questions.")

        final: list[QuizQuestion] = []
        for q in questions:
            page = q.source_page if q.source_page in valid_pages else None
            order = list(range(NUM_OPTIONS))
            self._rng.shuffle(order)
            final.append(
                QuizQuestion(
                    question=q.question,
                    options=[q.options[i] for i in order],
                    answer_index=order.index(q.answer_index),
                    explanation=q.explanation,
                    source_page=page,
                )
            )

        logger.info("Generated quiz with %d questions from %s", len(final), source_label)
        return Quiz(
            questions=final,
            source=source_label,
            pages=sorted(valid_pages),
            warnings=warnings,
        )


def _main(argv: list[str]) -> int:
    """Developer CLI: make a quiz from one PDF.

    Usage: python -m notemind.quiz path/to/file.pdf [num_questions] [start_page] [end_page]
    """
    from notemind.config import load_config, setup_logging
    from notemind.llm_client import LLMError, create_llm_client
    from notemind.pdf_loader import PDFLoader, PDFLoadError

    if not 1 <= len(argv) <= 4:
        print("Usage: python -m notemind.quiz file.pdf [num_questions] [start_page] [end_page]")
        return 2
    config = load_config()
    setup_logging(config.log_level)
    numbers = [int(x) for x in argv[1:]]
    n = numbers[0] if numbers else config.quiz.default_num_questions
    start = numbers[1] if len(numbers) > 1 else None
    end = numbers[2] if len(numbers) > 2 else None

    try:
        result = PDFLoader().load(argv[0])
        pages = filter_pages(result.pages, result.report.source, start, end)
        quiz = QuizGenerator(create_llm_client(config.generation), config.quiz).generate(pages, n)
    except (FileNotFoundError, PDFLoadError, QuizGenerationError, LLMError) as exc:
        print(f"ERROR ({type(exc).__name__}): {exc}")
        return 1

    for warning in quiz.warnings:
        print(f"WARNING: {warning}")
    for number, q in enumerate(quiz.questions, start=1):
        print(f"\nQ{number}. {q.question}" + (f"   [p.{q.source_page}]" if q.source_page else ""))
        for letter, option in zip("ABCD", q.options):
            print(f"   {letter}) {option}")
        print(f"   Answer: {'ABCD'[q.answer_index]}. {q.explanation}")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv[1:]))
