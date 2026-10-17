"""Tests for PDFLoader: page numbers, empty/scanned detection, bad files."""

from __future__ import annotations

from pathlib import Path

import pytest

from notemind.pdf_loader import PDFLoader, PDFLoadError, clean_text


def test_pages_keep_their_numbers_and_text(notes_pdf_bytes: bytes) -> None:
    result = PDFLoader().load_bytes(notes_pdf_bytes, "folder/ml.pdf")
    assert result.report.source == "ml.pdf"  # directories stripped for citations
    assert [p.page_number for p in result.pages] == [1, 2, 3]
    assert "learning rate" in result.pages[0].text
    assert "chain rule" in result.pages[1].text
    assert result.pages[2].is_empty


def test_empty_page_is_reported_but_pdf_is_not_scanned(notes_pdf_bytes: bytes) -> None:
    report = PDFLoader().load_bytes(notes_pdf_bytes, "ml.pdf").report
    assert report.empty_pages == [3]
    assert report.text_pages == 2
    assert not report.is_scanned
    assert any("no text" in w for w in report.warnings)


def test_image_only_pdf_is_flagged_as_scanned(make_pdf) -> None:
    report = PDFLoader().load_bytes(make_pdf(["", "", ""]), "scan.pdf").report
    assert report.is_scanned
    assert report.text_pages == 0
    assert "OCR" in report.warnings[0]


def test_mostly_empty_pdf_is_flagged_as_scanned(make_pdf) -> None:
    pages = ["Only this page has real extractable text on it."] + [""] * 9
    report = PDFLoader().load_bytes(make_pdf(pages), "mostly.pdf").report
    assert report.is_scanned
    assert report.text_pages == 1


def test_load_from_disk(tmp_path: Path, notes_pdf_bytes: bytes) -> None:
    path = tmp_path / "notes.pdf"
    path.write_bytes(notes_pdf_bytes)
    assert len(PDFLoader().load(path).text_pages) == 2
    with pytest.raises(FileNotFoundError):
        PDFLoader().load(tmp_path / "missing.pdf")


@pytest.mark.parametrize(
    "data,message",
    [
        (b"", "empty"),
        (b"Propositional logic questions", "not a real PDF"),
        (b"PK\x03\x04 zip content", "Word"),
        (b"%PDF-1.4 truncated garbage", "not a valid PDF"),
    ],
)
def test_bad_files_raise_clear_errors(data: bytes, message: str) -> None:
    with pytest.raises(PDFLoadError, match=message):
        PDFLoader().load_bytes(data, "bad.pdf")


def test_clean_text_normalises_whitespace_and_hyphenation() -> None:
    raw = "algo-\nrithm  uses\t\tarrays\n\n\n\nend\x00"
    assert clean_text(raw) == "algorithm uses arrays\n\nend"
