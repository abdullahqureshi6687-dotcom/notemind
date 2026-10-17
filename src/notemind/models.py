"""Plain data objects shared by every NoteMind component.

Keeping these as small frozen dataclasses (instead of passing dicts around)
gives type hints everywhere, makes objects hashable and safe to cache, and
gives one obvious place to define how a citation is formatted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Page:
    """Text extracted from one PDF page.

    Attributes:
        source: File name of the PDF (no directory), e.g. ``"lecture3.pdf"``.
        page_number: 1-based page number, matching what a PDF viewer shows.
        text: Cleaned text of the page (may be empty for image-only pages).
    """

    source: str
    page_number: int
    text: str

    @property
    def is_empty(self) -> bool:
        """True if the page has no extractable text."""
        return not self.text.strip()


@dataclass(frozen=True)
class Chunk:
    """A piece of one page's text, the unit that gets embedded and retrieved.

    A chunk never crosses a page boundary, so ``(source, page_number)`` is
    always an exact citation.

    Attributes:
        chunk_id: Stable, deterministic ID: ``"<source>::p<page>::c<index>"``.
        source: PDF file name.
        page_number: 1-based page the chunk comes from.
        chunk_index: Position of the chunk within its page (0-based).
        start_char: Character offset of the chunk within the page text.
        text: The chunk text.
    """

    chunk_id: str
    source: str
    page_number: int
    chunk_index: int
    start_char: int
    text: str

    @property
    def citation(self) -> str:
        """Citation in the format the generator is told to use."""
        return f"[{self.source}, p.{self.page_number}]"

    def to_dict(self) -> dict[str, Any]:
        """Serialise for saving an index to disk."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        """Rebuild a chunk saved with :meth:`to_dict`."""
        return cls(**data)


@dataclass(frozen=True)
class SearchResult:
    """One retrieved chunk with its score.

    Attributes:
        chunk: The retrieved chunk.
        score: Retriever-specific score. Cosine similarity for dense search,
            BM25 score for keyword search, RRF score for hybrid. Scores from
            different methods are NOT comparable with each other.
        rank: 1-based position in the result list.
        method: Which retriever produced it: heap, faiss, bm25 or hybrid.
    """

    chunk: Chunk
    score: float
    rank: int
    method: str


@dataclass
class LoadReport:
    """What happened while loading one PDF, shown to the user as warnings.

    Attributes:
        source: PDF file name.
        num_pages: Total pages in the file.
        empty_pages: 1-based numbers of pages with no extractable text.
        is_scanned: True if the PDF looks image-only (needs OCR, unsupported).
        warnings: Human-readable messages for the UI.
    """

    source: str
    num_pages: int = 0
    empty_pages: list[int] = field(default_factory=list)
    is_scanned: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def text_pages(self) -> int:
        """Number of pages that produced text."""
        return self.num_pages - len(self.empty_pages)


@dataclass
class Answer:
    """The final response to a user question.

    Attributes:
        text: The answer shown in the chat (includes inline citations).
        sources: Chunks that were given to the LLM as context.
        refused: True if the system answered "not in your notes".
        refusal_reason: Why it refused: ``"low_similarity"`` (no relevant
            chunk retrieved, LLM not called), ``"no_index"``, ``"llm"`` (the
            model decided the context was insufficient) or ``None``.
        model: Model that produced the answer, if one was called.
        invalid_citations: Citations in the answer that do not match any
            retrieved chunk (a sign the model invented a source).
        max_similarity: Best cosine score from retrieval, for display/debugging.
    """

    text: str
    sources: list[SearchResult] = field(default_factory=list)
    refused: bool = False
    refusal_reason: str | None = None
    model: str | None = None
    invalid_citations: list[str] = field(default_factory=list)
    max_similarity: float | None = None
