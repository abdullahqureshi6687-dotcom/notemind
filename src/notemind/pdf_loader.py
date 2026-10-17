"""PDF ingestion that keeps page numbers.

Text is extracted one page at a time with pypdf so every piece of text keeps
the page it came from. Image-only (scanned) PDFs are detected and reported
with a clear warning instead of silently producing an empty index; OCR is out
of scope.

Run directly to inspect a PDF:  ``python -m notemind.pdf_loader path/to/file.pdf``
"""

from __future__ import annotations

import io
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from notemind.models import LoadReport, Page

logger = logging.getLogger(__name__)

# A page with fewer extractable characters than this is treated as empty
# (it is usually just a page number, a logo or a scanned image).
MIN_PAGE_CHARS: int = 20
# If at least this fraction of pages is empty, the PDF is reported as scanned.
SCANNED_EMPTY_RATIO: float = 0.8

_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")          # "algo-\nrithm" -> "algorithm"
_SPACES = re.compile(r"[ \t\f\v ]+")            # runs of horizontal whitespace
_MANY_NEWLINES = re.compile(r"\n{3,}")               # keep at most one blank line
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")  # stray control characters


class PDFLoadError(Exception):
    """Raised when a file cannot be read as a PDF at all."""


@dataclass
class LoadResult:
    """Pages from one PDF plus a report of anything that went wrong."""

    pages: list[Page] = field(default_factory=list)
    report: LoadReport = field(default_factory=lambda: LoadReport(source=""))

    @property
    def text_pages(self) -> list[Page]:
        """Only the pages that contain text."""
        return [p for p in self.pages if not p.is_empty]


def clean_text(text: str) -> str:
    """Normalise raw extracted text without changing its meaning.

    Joins words hyphenated across line breaks, collapses repeated spaces,
    removes control characters and limits consecutive blank lines.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip()


class PDFLoader:
    """Extract per-page text from PDF files or uploaded bytes.

    Example::

        result = PDFLoader().load("data/sample_pdfs/lecture1.pdf")
        for warning in result.report.warnings:
            print(warning)
        pages = result.text_pages
    """

    def __init__(
        self,
        min_page_chars: int = MIN_PAGE_CHARS,
        scanned_empty_ratio: float = SCANNED_EMPTY_RATIO,
    ) -> None:
        """
        Args:
            min_page_chars: Pages with fewer characters are counted as empty.
            scanned_empty_ratio: Fraction of empty pages at which the whole
                PDF is reported as scanned / image-only.
        """
        self.min_page_chars = min_page_chars
        self.scanned_empty_ratio = scanned_empty_ratio

    def load(self, path: str | Path) -> LoadResult:
        """Load a PDF from disk.

        Raises:
            FileNotFoundError: If the path does not exist.
            PDFLoadError: If the file is not a readable PDF.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"PDF not found: {path}")
        return self.load_bytes(path.read_bytes(), path.name)

    def load_bytes(self, data: bytes, filename: str) -> LoadResult:
        """Load a PDF from raw bytes (what Streamlit's file uploader gives you).

        Args:
            data: The PDF file content.
            filename: Name used in citations. Directories are stripped.

        Raises:
            PDFLoadError: If the bytes are empty or not a readable PDF.
        """
        source = Path(filename).name
        if not data:
            raise PDFLoadError(f"{source}: file is empty")

        # Real PDFs contain the "%PDF" signature in their first bytes. Checking
        # it gives a clear message for renamed text/Word files instead of a
        # cryptic parser error.
        if b"%PDF" not in data[:1024]:
            start = data[:20].decode("utf-8", errors="replace").strip()
            hint = " (looks like a Word .docx file)" if data[:2] == b"PK" else ""
            raise PDFLoadError(
                f"{source}: not a real PDF{hint}. The file starts with {start!r} instead of "
                "'%PDF', so it was probably renamed to .pdf rather than exported. "
                "Open it and use 'Save as PDF' or 'Print to PDF'."
            )

        try:
            reader = PdfReader(io.BytesIO(data))
        except (PdfReadError, ValueError, OSError) as exc:
            raise PDFLoadError(f"{source}: not a valid PDF ({exc})") from exc

        if reader.is_encrypted:
            # Many lecture PDFs are "encrypted" with an empty user password,
            # which only restricts printing/copying. Try that before giving up.
            try:
                if not reader.decrypt(""):
                    raise PDFLoadError(f"{source}: PDF is password-protected")
            except PDFLoadError:
                raise
            except Exception as exc:  # pypdf raises several types here
                raise PDFLoadError(f"{source}: could not decrypt PDF ({exc})") from exc

        report = LoadReport(source=source, num_pages=len(reader.pages))
        pages: list[Page] = []

        for index, pdf_page in enumerate(reader.pages):
            page_number = index + 1
            try:
                raw = pdf_page.extract_text() or ""
            except Exception as exc:  # one broken page should not kill the file
                logger.warning("%s p.%d: text extraction failed: %s", source, page_number, exc)
                report.warnings.append(f"Page {page_number}: text extraction failed, skipped.")
                raw = ""

            text = clean_text(raw)
            if len(text) < self.min_page_chars:
                report.empty_pages.append(page_number)
                text = ""
            pages.append(Page(source=source, page_number=page_number, text=text))

        self._add_summary_warnings(report)
        logger.info(
            "Loaded %s: %d pages, %d with text%s",
            source,
            report.num_pages,
            report.text_pages,
            " (looks scanned)" if report.is_scanned else "",
        )
        return LoadResult(pages=pages, report=report)

    def _add_summary_warnings(self, report: LoadReport) -> None:
        """Decide whether the PDF looks scanned and add user-facing warnings."""
        if report.num_pages == 0:
            report.warnings.append("This PDF has no pages.")
            report.is_scanned = False
            return

        empty_ratio = len(report.empty_pages) / report.num_pages
        if report.text_pages == 0:
            report.is_scanned = True
            report.warnings.append(
                "No text could be extracted. This looks like a scanned or image-only "
                "PDF; OCR is not supported, so it cannot be searched."
            )
        elif empty_ratio >= self.scanned_empty_ratio:
            report.is_scanned = True
            report.warnings.append(
                f"Only {report.text_pages} of {report.num_pages} pages contain text. "
                "This PDF is probably mostly scanned images; most of it cannot be searched."
            )
        elif report.empty_pages:
            shown = ", ".join(str(p) for p in report.empty_pages[:10])
            more = " ..." if len(report.empty_pages) > 10 else ""
            report.warnings.append(
                f"{len(report.empty_pages)} page(s) had no text and were skipped: {shown}{more}"
            )


def _main(argv: list[str]) -> int:
    """Developer CLI: show what the loader and chunker produce for a PDF."""
    from notemind.chunker import Chunker
    from notemind.config import load_config, setup_logging

    if len(argv) != 1:
        print("Usage: python -m notemind.pdf_loader path/to/file.pdf")
        return 2

    config = load_config()
    setup_logging(config.log_level)
    try:
        result = PDFLoader().load(argv[0])
    except (FileNotFoundError, PDFLoadError) as exc:
        print(f"ERROR: {exc}")
        return 1

    report = result.report
    print(f"\nFile:          {report.source}")
    print(f"Pages:         {report.num_pages} ({report.text_pages} with text)")
    print(f"Looks scanned: {report.is_scanned}")
    for warning in report.warnings:
        print(f"WARNING:       {warning}")

    chunks = Chunker(config.chunking).chunk_pages(result.pages)
    print(f"Chunks:        {len(chunks)} (size={config.chunking.chunk_size}, "
          f"overlap={config.chunking.chunk_overlap})")
    for chunk in chunks[:3]:
        preview = chunk.text[:150].replace("\n", " ")
        print(f"\n{chunk.chunk_id} {chunk.citation} ({len(chunk.text)} chars)\n  {preview}...")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
