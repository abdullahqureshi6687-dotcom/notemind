"""Shared test fixtures.

Every test runs offline in a few seconds: the embedding model and the LLM are
replaced by the test doubles in ``fakes.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fakes import HashingEncoder, build_pdf

from notemind.bm25_index import BM25Index
from notemind.chunker import Chunker
from notemind.config import AppConfig, ChunkingConfig, RetrievalConfig, load_config
from notemind.hybrid import HybridRetriever
from notemind.models import Chunk, Page
from notemind.vector_store import VectorStore

# Small "lecture notes" corpus: one clear topic per page, so each test
# question has exactly one correct page.
NOTES: dict[tuple[str, int], str] = {
    ("ml.pdf", 1): "Gradient descent minimises a loss function by moving the parameters "
    "against the gradient. The learning rate controls the size of each step.",
    ("ml.pdf", 2): "Backpropagation computes the gradient of the loss with respect to every "
    "weight by applying the chain rule backwards through the network layers.",
    ("ml.pdf", 3): "Overfitting happens when a model memorises training data. Regularisation "
    "such as dropout or L2 weight decay reduces overfitting.",
    ("dsa.pdf", 1): "A binary heap is a complete binary tree stored in an array. The children "
    "of the node at index i are stored at indices 2i+1 and 2i+2.",
    ("dsa.pdf", 2): "Dijkstra's algorithm finds shortest paths in a graph with non-negative edge "
    "weights using a priority queue.",
}


@pytest.fixture
def encoder() -> HashingEncoder:
    return HashingEncoder()


@pytest.fixture
def pages() -> list[Page]:
    return [Page(source, number, text) for (source, number), text in NOTES.items()]


@pytest.fixture
def chunks(pages: list[Page]) -> list[Chunk]:
    return Chunker(ChunkingConfig(chunk_size=300, chunk_overlap=50, min_chunk_chars=20)).chunk_pages(pages)


@pytest.fixture
def store(chunks: list[Chunk], encoder: HashingEncoder) -> VectorStore:
    vector_store = VectorStore(encoder.dimension)
    vector_store.add(chunks, encoder.encode([c.text for c in chunks]))
    return vector_store


@pytest.fixture
def retriever(store: VectorStore, chunks: list[Chunk], encoder: HashingEncoder) -> HybridRetriever:
    return HybridRetriever(store, BM25Index(chunks), encoder, RetrievalConfig(top_k=3, candidate_k=5))


@pytest.fixture
def app_config(tmp_path: Path) -> AppConfig:
    """Project config with the index cache redirected to a temp folder."""
    return load_config().with_overrides(paths={"index_dir": tmp_path / "indexes"})


@pytest.fixture
def make_pdf():
    """The build_pdf helper, as a fixture (avoids importing conftest directly)."""
    return build_pdf


@pytest.fixture
def notes_pdf_bytes() -> bytes:
    """A 3-page PDF: two pages of text and a blank third page."""
    return build_pdf([NOTES[("ml.pdf", 1)], NOTES[("ml.pdf", 2)], ""])
