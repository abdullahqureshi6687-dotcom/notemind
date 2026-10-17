"""Tests for VectorStore (heap + FAISS), BM25Index, RRF and HybridRetriever."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from notemind.bm25_index import BM25Index, tokenize
from notemind.hybrid import HybridRetriever, reciprocal_rank_fusion
from notemind.models import Chunk, SearchResult
from notemind.vector_store import IndexLoadError, VectorStore


def make_chunk(cid: str, text: str = "text", source: str = "f.pdf", page: int = 1) -> Chunk:
    return Chunk(chunk_id=cid, source=source, page_number=page, chunk_index=0, start_char=0, text=text)


def random_store(n: int, dim: int, seed: int) -> VectorStore:
    rng = np.random.default_rng(seed)
    store = VectorStore(dim)
    store.add([make_chunk(f"f::p1::c{i}") for i in range(n)], rng.standard_normal((n, dim)))
    return store


# ----------------------------------------------------------------- VectorStore


@pytest.mark.parametrize("seed", range(15))
def test_heap_and_faiss_return_identical_results(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n, k = int(rng.integers(1, 400)), int(rng.integers(1, 20))
    store = random_store(n, 32, seed)
    query = rng.standard_normal(32)

    heap = store.search(query, k, backend="heap")
    faiss = store.search(query, k, backend="faiss")
    assert [r.chunk.chunk_id for r in heap] == [r.chunk.chunk_id for r in faiss]
    assert np.allclose([r.score for r in heap], [r.score for r in faiss], atol=1e-5)
    assert [r.rank for r in heap] == list(range(1, min(n, k) + 1))


def test_search_scores_are_cosine_similarities() -> None:
    store = VectorStore(2)
    store.add([make_chunk("a"), make_chunk("b")], np.array([[3.0, 0.0], [0.0, 5.0]]))
    results = store.search(np.array([2.0, 0.0]), 2)
    assert results[0].chunk.chunk_id == "a"
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.0, abs=1e-6)


def test_empty_store_and_invalid_backend() -> None:
    store = VectorStore(4)
    assert store.search(np.ones(4), 3) == []
    with pytest.raises(ValueError):
        random_store(3, 4, 0).search(np.ones(4), 1, backend="annoy")


def test_add_validates_shapes_and_duplicate_ids() -> None:
    store = VectorStore(4)
    with pytest.raises(ValueError):
        store.add([make_chunk("a")], np.ones((1, 3)))  # wrong dimension
    with pytest.raises(ValueError):
        store.add([make_chunk("a"), make_chunk("b")], np.ones((1, 4)))  # count mismatch
    store.add([make_chunk("a")], np.ones((1, 4)))
    with pytest.raises(ValueError):
        store.add([make_chunk("a")], np.ones((1, 4)))  # duplicate ID


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    store = random_store(50, 16, seed=1)
    store.save(tmp_path / "idx", metadata={"key": "abc"})
    loaded, metadata = VectorStore.load(tmp_path / "idx")

    assert metadata == {"key": "abc"}
    assert [c.chunk_id for c in loaded.chunks] == [c.chunk_id for c in store.chunks]
    query = np.random.default_rng(2).standard_normal(16)
    for backend in ("heap", "faiss"):
        assert [r.chunk.chunk_id for r in loaded.search(query, 5, backend)] == [
            r.chunk.chunk_id for r in store.search(query, 5, backend)
        ]


def test_load_rejects_missing_or_inconsistent_index(tmp_path: Path) -> None:
    with pytest.raises(IndexLoadError):
        VectorStore.load(tmp_path / "nothing-here")

    folder = random_store(5, 8, seed=3).save(tmp_path / "idx")
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest["num_chunks"] = 99
    (folder / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(IndexLoadError, match="disagree"):
        VectorStore.load(folder)


# ----------------------------------------------------------------- BM25


def test_tokenize_lowercases_drops_stopwords_and_splits_joined_terms() -> None:
    assert tokenize("What is the K-means algorithm?") == ["k-means", "k", "means", "algorithm"]


def test_bm25_ranks_the_chunk_with_the_rare_term_first(chunks: list[Chunk]) -> None:
    results = BM25Index(chunks).search("Dijkstra shortest paths", k=3)
    assert results[0].chunk.citation == "[dsa.pdf, p.2]"
    assert all(r.method == "bm25" for r in results)


def test_bm25_returns_only_chunks_sharing_a_query_term(chunks: list[Chunk]) -> None:
    index = BM25Index(chunks)
    assert index.search("photosynthesis chloroplast", k=5) == []
    assert index.search("what is the", k=5) == []  # stopwords only
    assert index.search("dropout", k=5)[0].chunk.citation == "[ml.pdf, p.3]"
    assert len(index.search("dropout", k=5)) == 1


def test_bm25_on_empty_corpus() -> None:
    assert BM25Index([]).search("anything", 3) == []


# ----------------------------------------------------------------- RRF


def test_rrf_scores_match_the_formula() -> None:
    a, b, c = make_chunk("A"), make_chunk("B"), make_chunk("C")
    dense = [SearchResult(a, 0.9, 1, "faiss"), SearchResult(b, 0.8, 2, "faiss")]
    sparse = [SearchResult(b, 5.0, 1, "bm25"), SearchResult(c, 4.0, 2, "bm25"), SearchResult(a, 3.0, 3, "bm25")]
    fused = reciprocal_rank_fusion([dense, sparse], k=60)

    assert [r.chunk.chunk_id for r in fused] == ["B", "A", "C"]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[1].score == pytest.approx(1 / 61 + 1 / 63)
    assert fused[2].score == pytest.approx(1 / 62)
    assert [r.rank for r in fused] == [1, 2, 3]
    assert all(r.method == "hybrid" for r in fused)


def test_rrf_top_n_ties_and_validation() -> None:
    a, b = make_chunk("A"), make_chunk("B")
    fused = reciprocal_rank_fusion([[SearchResult(b, 1, 1, "x")], [SearchResult(a, 1, 1, "y")]], top_n=1)
    assert [r.chunk.chunk_id for r in fused] == ["A"]  # equal score and rank: sorted by ID
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], k=0)


# ----------------------------------------------------------------- HybridRetriever


@pytest.mark.parametrize("mode", ["heap", "faiss", "bm25", "hybrid"])
@pytest.mark.parametrize(
    "question,expected",
    [
        ("What does the learning rate control in gradient descent?", "[ml.pdf, p.1]"),
        ("How does backpropagation use the chain rule?", "[ml.pdf, p.2]"),
        ("Where are the children of node i stored in a binary heap array?", "[dsa.pdf, p.1]"),
    ],
)
def test_every_mode_finds_the_right_page(
    retriever: HybridRetriever, mode: str, question: str, expected: str
) -> None:
    output = retriever.retrieve(question, mode=mode)
    assert output.mode == mode
    assert 1 <= len(output.results) <= 3
    assert output.results[0].chunk.citation == expected
    assert output.max_similarity is not None and output.max_similarity > 0.2


def test_off_topic_question_has_low_similarity(retriever: HybridRetriever) -> None:
    output = retriever.retrieve("photosynthesis in chloroplasts of green plants")
    assert output.max_similarity is not None
    assert output.max_similarity < 0.25


def test_top_k_override_and_empty_query(retriever: HybridRetriever) -> None:
    assert len(retriever.retrieve("gradient", top_k=1).results) == 1
    empty = retriever.retrieve("   ")
    assert empty.results == [] and empty.max_similarity is None
    with pytest.raises(ValueError):
        retriever.retrieve("gradient", mode="semantic")


def test_retriever_rejects_mismatched_indexes(store: VectorStore, chunks: list[Chunk], encoder) -> None:
    with pytest.raises(ValueError):
        HybridRetriever(store, BM25Index(chunks[:1]), encoder)
