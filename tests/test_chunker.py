"""Tests for Chunker: size limits, overlap, coverage, page boundaries, IDs."""

from __future__ import annotations

import random

import pytest

from notemind.chunker import Chunker
from notemind.config import ChunkingConfig, ConfigError
from notemind.models import Page

LONG_TEXT = " ".join(
    f"Sentence number {i} explains a concept from the lecture in some detail." for i in range(60)
)


def make_chunker(size: int, overlap: int, min_chars: int = 0) -> Chunker:
    return Chunker(ChunkingConfig(chunk_size=size, chunk_overlap=overlap, min_chunk_chars=min_chars))


@pytest.mark.parametrize("size,overlap", [(100, 0), (200, 50), (300, 100), (500, 150), (800, 799)])
def test_chunks_never_exceed_chunk_size(size: int, overlap: int) -> None:
    pieces = make_chunker(size, overlap).split_text(LONG_TEXT)
    assert pieces, "expected at least one chunk"
    assert all(len(text) <= size for _, text in pieces)


def test_offsets_point_at_the_chunk_text() -> None:
    text = LONG_TEXT.strip()
    for offset, piece in make_chunker(200, 50).split_text(text):
        assert text[offset : offset + len(piece)] == piece


def test_consecutive_chunks_overlap() -> None:
    pieces = make_chunker(200, 60).split_text(LONG_TEXT)
    for (start_a, text_a), (start_b, _) in zip(pieces, pieces[1:]):
        end_a = start_a + len(text_a)
        assert start_b < end_a, "next chunk should start inside the previous one"
        assert start_b > start_a, "chunks must always move forward"


def test_zero_overlap_produces_no_shared_text() -> None:
    pieces = make_chunker(200, 0).split_text(LONG_TEXT)
    for (start_a, text_a), (start_b, _) in zip(pieces, pieces[1:]):
        assert start_b >= start_a + len(text_a)


@pytest.mark.parametrize("seed", range(20))
def test_no_text_is_lost(seed: int) -> None:
    """Every non-space character is covered by at least one chunk."""
    rng = random.Random(seed)
    words = ["gradient", "loss.", "heap", "tree?", "array", "rule!", "x" * 40, "\n\n", "network"]
    text = " ".join(rng.choice(words) for _ in range(rng.randint(1, 400))).strip()
    size = rng.choice([100, 150, 300, 500])
    chunker = make_chunker(size, rng.randint(0, size - 1))

    covered = [False] * len(text)
    for offset, piece in chunker.split_text(text):
        for i in range(offset, offset + len(piece)):
            covered[i] = True
    missing = [i for i, ch in enumerate(text) if not covered[i] and not ch.isspace()]
    assert not missing


def test_text_without_spaces_terminates_and_is_split() -> None:
    pieces = make_chunker(100, 30).split_text("x" * 1000)
    assert len(pieces) >= 10
    assert all(len(p) <= 100 for _, p in pieces)


def test_prefers_ending_at_a_sentence_boundary() -> None:
    pieces = make_chunker(200, 0).split_text(LONG_TEXT)
    assert all(text.endswith(".") for _, text in pieces[:-1])


def test_next_chunk_starts_at_a_word_boundary() -> None:
    text = LONG_TEXT.strip()
    for offset, _ in make_chunker(150, 40).split_text(text)[1:]:
        assert offset == 0 or text[offset - 1].isspace()


def test_chunks_never_cross_page_boundaries() -> None:
    pages = [Page("a.pdf", 1, LONG_TEXT), Page("a.pdf", 2, "Short second page about heaps and trees.")]
    chunks = make_chunker(200, 50).chunk_pages(pages)
    page_two = [c for c in chunks if c.page_number == 2]
    assert len(page_two) == 1
    assert page_two[0].text == pages[1].text
    for chunk in chunks:
        assert chunk.text in pages[chunk.page_number - 1].text


def test_chunk_ids_are_unique_deterministic_and_cite_correctly() -> None:
    pages = [Page("lec 1.pdf", 3, LONG_TEXT), Page("lec 1.pdf", 4, LONG_TEXT)]
    first = make_chunker(300, 50).chunk_pages(pages)
    second = make_chunker(300, 50).chunk_pages(pages)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert len({c.chunk_id for c in first}) == len(first)
    assert first[0].chunk_id == "lec 1.pdf::p3::c0"
    assert first[0].citation == "[lec 1.pdf, p.3]"


def test_empty_pages_and_short_fragments_are_skipped() -> None:
    pages = [Page("a.pdf", 1, ""), Page("a.pdf", 2, "  12  "), Page("a.pdf", 3, LONG_TEXT)]
    chunks = make_chunker(300, 50, min_chars=40).chunk_pages(pages)
    assert chunks
    assert {c.page_number for c in chunks} == {3}
    assert all(len(c.text) >= 40 for c in chunks)


def test_empty_text_gives_no_chunks() -> None:
    assert make_chunker(200, 50).split_text("") == []
    assert make_chunker(200, 50).split_text("   \n  ") == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_size": 200, "chunk_overlap": 200},
        {"chunk_size": 200, "chunk_overlap": 250},
        {"chunk_size": 50, "chunk_overlap": 0},
        {"chunk_size": 200, "chunk_overlap": -1},
    ],
)
def test_invalid_chunking_config_is_rejected(kwargs: dict) -> None:
    with pytest.raises(ConfigError):
        ChunkingConfig(**kwargs)
