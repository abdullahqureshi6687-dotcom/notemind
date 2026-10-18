"""Grounded answer generation with citation checking.

The Generator turns retrieved chunks into an :class:`~notemind.models.Answer`:

1. No retrieved chunks -> refuse immediately (no LLM call, no cost).
2. Build a labelled context, capped at ``max_context_chars``.
3. Ask the LLM, with the grounding rules from :mod:`notemind.prompts`.
4. Detect the refusal sentence, so refusals can be shown and measured.
5. Verify citations: every ``[file, p.X]`` in the answer must match a chunk
   that was actually in the context. Invented citations are reported.

Step 5 matters because "the model said it's on page 7" is only useful if
page 7 was really retrieved. It turns a vague "hallucination" concern into a
measurable check.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from notemind.config import GenerationConfig
from notemind.llm_client import LLMClient
from notemind.models import Answer, SearchResult
from notemind.prompts import (
    ANSWER_SYSTEM_PROMPT,
    REFUSAL_TEXT,
    build_answer_prompt,
    format_context,
)

logger = logging.getLogger(__name__)

# Matches [lecture3.pdf, p.12], [lecture3.pdf, p. 12], [lecture3.pdf, page 12]
CITATION_PATTERN = re.compile(
    r"\[\s*([^\[\],]+?\.pdf)\s*,\s*p(?:age|g)?\.?\s*(\d+)\s*\]",
    flags=re.IGNORECASE,
)

_REFUSAL_KEY = "couldn't find this in your notes"


def extract_citations(text: str) -> list[tuple[str, int]]:
    """All ``(file_name, page)`` citations in ``text``, in order, without duplicates."""
    seen: dict[tuple[str, int], None] = {}
    for match in CITATION_PATTERN.finditer(text):
        seen[(match.group(1).strip(), int(match.group(2)))] = None
    return list(seen)


def is_refusal(text: str) -> bool:
    """True if the model gave the "not in your notes" answer.

    Accepts a straight or curly apostrophe and minor variations, but only when
    the refusal is the main content (short answer or the opening sentence), so
    a partial answer that mentions a gap is not counted as a refusal.
    """
    normalized = text.strip().lower().replace("’", "'")
    if _REFUSAL_KEY not in normalized:
        return False
    return normalized.startswith("i couldn't find") or len(normalized) <= len(REFUSAL_TEXT) + 40


class Generator:
    """Answer questions from retrieved chunks using an LLM.

    Example::

        generator = Generator(create_llm_client(cfg.generation), cfg.generation)
        answer = generator.answer("What is backpropagation?", retrieval.results)
        print(answer.text, answer.invalid_citations)
    """

    def __init__(self, llm: LLMClient, config: GenerationConfig) -> None:
        self.llm = llm
        self.config = config

    def answer(self, question: str, results: Sequence[SearchResult]) -> Answer:
        """Generate a grounded answer.

        Args:
            question: The user's question.
            results: Retrieved chunks, best first.

        Returns:
            An :class:`Answer`. ``sources`` lists only the chunks that were
            actually sent to the model.

        Raises:
            LLMError: If the provider call fails after retries. The caller
                (pipeline/UI) decides how to show this to the user.
        """
        if not question.strip():
            return Answer(text="Please type a question.", refused=True, refusal_reason="empty_question")
        if not results:
            return Answer(text=REFUSAL_TEXT, refused=True, refusal_reason="no_results")

        context, used = format_context(results, self.config.max_context_chars)
        prompt = build_answer_prompt(question, context)
        text = self.llm.generate(ANSWER_SYSTEM_PROMPT, prompt)

        if is_refusal(text):
            logger.info("Model refused: context judged insufficient")
            return Answer(
                text=REFUSAL_TEXT,
                sources=used,
                refused=True,
                refusal_reason="llm",
                model=self.llm.model,
            )

        allowed = {(r.chunk.source.lower(), r.chunk.page_number) for r in used}
        invalid = [
            f"[{source}, p.{page}]"
            for source, page in extract_citations(text)
            if (source.lower(), page) not in allowed
        ]
        if invalid:
            logger.warning("Answer cites sources that were not retrieved: %s", invalid)
        if not extract_citations(text):
            logger.warning("Answer contains no citations")

        return Answer(
            text=text,
            sources=used,
            refused=False,
            model=self.llm.model,
            invalid_citations=invalid,
        )


def _main(argv: list[str]) -> int:
    """Developer CLI: answer one question from the PDFs in data_dir.

    Usage: python -m notemind.generator "your question"
    """
    from notemind.config import load_config, setup_logging
    from notemind.hybrid import build_demo_retriever
    from notemind.llm_client import LLMError, create_llm_client

    if len(argv) != 1:
        print('Usage: python -m notemind.generator "your question"')
        return 2
    config = load_config()
    setup_logging(config.log_level)
    try:
        llm = create_llm_client(config.generation)
    except LLMError as exc:
        print(f"ERROR: {exc}")
        return 1

    retriever = build_demo_retriever(config)
    if retriever is None:
        return 1
    retrieval = retriever.retrieve(argv[0])
    best = retrieval.max_similarity or 0.0
    print(f"Mode: {retrieval.mode}   best cosine: {best:.3f}   "
          f"threshold: {config.retrieval.min_similarity}")

    if best < config.retrieval.min_similarity:
        print(f"\n{REFUSAL_TEXT}  (below threshold, LLM not called)")
        return 0
    try:
        answer = Generator(llm, config.generation).answer(argv[0], retrieval.results)
    except LLMError as exc:
        print(f"ERROR ({type(exc).__name__}): {exc}")
        return 1

    print(f"\n{answer.text}\n")
    print(f"Refused: {answer.refused}   Model: {answer.model}")
    print("Sources sent to the model:", ", ".join(r.chunk.citation for r in answer.sources))
    if answer.invalid_citations:
        print("WARNING - citations not in the retrieved sources:", answer.invalid_citations)
    return 0


if __name__ == "__main__":
    import sys

    from notemind.config import run_cli

    raise SystemExit(run_cli(lambda: _main(sys.argv[1:])))
