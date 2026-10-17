"""Split page text into overlapping chunks.

Design decisions (worth explaining in an interview):

* **Chunks never cross page boundaries.** Each chunk belongs to exactly one
  page, so citations are exact. The cost is that a sentence split across two
  pages ends up in two chunks; overlap inside a page does not help with that.
* **Sizes are in characters**, which is simple and predictable. The config
  warns when chunk_size exceeds what all-MiniLM-L6-v2 can embed (~256 tokens).
* **Boundaries are snapped to natural breaks.** A chunk prefers to end at a
  sentence end or newline, then at a space, and only cuts mid-word if a single
  "word" is longer than half a chunk. The next chunk starts at a word boundary.
* **Chunk IDs are deterministic** (``file::p3::c1``), so the same PDF with the
  same settings always produces the same IDs. The eval script and the saved
  index both rely on this.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from notemind.config import ChunkingConfig
from notemind.models import Chunk, Page

logger = logging.getLogger(__name__)

# Characters after which a chunk may end "cleanly", in order of preference.
_SENTENCE_ENDS: tuple[str, ...] = (". ", "? ", "! ", ".\n", "?\n", "!\n", "\n\n")


class Chunker:
    """Turn pages into overlapping, page-local chunks.

    Example::

        chunker = Chunker(ChunkingConfig(chunk_size=500, chunk_overlap=100))
        chunks = chunker.chunk_pages(pages)
    """

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        """
        Args:
            config: Chunk size, overlap and minimum chunk length. Validation
                (overlap < size, etc.) happens inside :class:`ChunkingConfig`.
        """
        self.config = config or ChunkingConfig()

    @property
    def chunk_size(self) -> int:
        return self.config.chunk_size

    @property
    def chunk_overlap(self) -> int:
        return self.config.chunk_overlap

    def chunk_pages(self, pages: Iterable[Page]) -> list[Chunk]:
        """Chunk every page, skipping empty ones.

        Args:
            pages: Pages from :class:`~notemind.pdf_loader.PDFLoader`.

        Returns:
            Chunks in document order (by page, then position on the page).
        """
        chunks: list[Chunk] = []
        for page in pages:
            if page.is_empty:
                continue
            chunks.extend(self.chunk_page(page))
        logger.debug("Created %d chunks", len(chunks))
        return chunks

    def chunk_page(self, page: Page) -> list[Chunk]:
        """Chunk a single page."""
        return [
            Chunk(
                chunk_id=f"{page.source}::p{page.page_number}::c{index}",
                source=page.source,
                page_number=page.page_number,
                chunk_index=index,
                start_char=start,
                text=text,
            )
            for index, (start, text) in enumerate(self.split_text(page.text))
        ]

    def split_text(self, text: str) -> list[tuple[int, str]]:
        """Split text into ``(start_offset, chunk_text)`` pairs.

        The loop always moves forward, so it terminates for any input,
        including text with no spaces at all.
        """
        text = text.strip()
        n = len(text)
        size, overlap = self.chunk_size, self.chunk_overlap
        pieces: list[tuple[int, str]] = []
        start = 0

        while start < n:
            end = min(start + size, n)
            if end < n:
                end = self._find_break(text, start, end)

            piece = text[start:end].strip()
            if len(piece) >= self.config.min_chunk_chars:
                # Offset of the stripped piece, so text[offset:] starts with it.
                offset = start + (len(text[start:end]) - len(text[start:end].lstrip()))
                pieces.append((offset, piece))

            if end >= n:
                break

            next_start = self._snap_to_word_start(text, end - overlap, end)
            if next_start <= start:  # guarantee progress
                next_start = end
            start = next_start

        return pieces

    def _find_break(self, text: str, start: int, end: int) -> int:
        """Best place to end a chunk inside ``text[start:end]``.

        Looks only in the second half of the window so chunks don't become
        tiny. Prefers sentence ends, then any whitespace, else a hard cut.
        """
        search_from = start + (end - start) // 2

        best = -1
        for marker in _SENTENCE_ENDS:
            pos = text.rfind(marker, search_from, end)
            if pos != -1:
                best = max(best, pos + len(marker.rstrip(" ")))
        if best > search_from:
            return best

        for i in range(end - 1, search_from, -1):
            if text[i].isspace():
                return i
        return end

    @staticmethod
    def _snap_to_word_start(text: str, pos: int, limit: int) -> int:
        """Move ``pos`` forward to the start of the next word (before ``limit``).

        Prevents the overlap from beginning in the middle of a word.
        """
        pos = max(pos, 0)
        if pos == 0 or text[pos - 1].isspace():
            return pos
        for i in range(pos, limit):
            if text[i].isspace():
                return i + 1
        return pos
