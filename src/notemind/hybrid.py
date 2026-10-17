"""One retrieval entry point for all four search modes.

Modes (``retrieval.mode`` in config.yaml):

* ``heap``   - dense search with the from-scratch heap top-k
* ``faiss``  - dense search with FAISS
* ``bm25``   - keyword search
* ``hybrid`` - dense + BM25 combined with Reciprocal Rank Fusion (RRF)

**Why RRF instead of adding scores?** Cosine similarities live in [-1, 1]
and cluster around 0.2-0.7; BM25 scores are unbounded and depend on corpus
size. Adding them (even after min-max scaling) lets one retriever dominate
unpredictably. RRF uses only *ranks*:

    rrf(d) = sum over retrievers r of 1 / (k + rank_r(d))

A chunk ranked highly by both retrievers wins; a chunk found by only one can
still make the cut. k (default 60) dampens the gap between rank 1 and rank 2.

**Refusal signal.** Every query also records the best dense cosine score,
even in BM25 mode. The pipeline compares it with ``min_similarity`` to answer
"I couldn't find this in your notes" without calling the LLM. RRF scores can't
be used for this because they say nothing about absolute relevance.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from notemind.bm25_index import BM25Index
from notemind.config import RetrievalConfig
from notemind.embedder import TextEncoder
from notemind.models import Chunk, SearchResult
from notemind.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalOutput:
    """Ranked results plus the signal used to decide whether to refuse.

    Attributes:
        results: Final top-k results for the chosen mode.
        max_similarity: Highest cosine similarity between the query and any
            chunk (None if the store is empty).
        mode: The mode that produced ``results``.
    """

    results: list[SearchResult] = field(default_factory=list)
    max_similarity: float | None = None
    mode: str = ""


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[SearchResult]],
    k: int = 60,
    top_n: int | None = None,
) -> list[SearchResult]:
    """Fuse several ranked lists with RRF.

    Args:
        ranked_lists: Each list sorted best-first (ranks start at 1).
        k: RRF constant. Larger values flatten the advantage of top ranks.
        top_n: Keep only the best ``top_n`` fused results (None = all).

    Returns:
        Fused results with ``method="hybrid"`` and the RRF score as ``score``.
        Ties are broken by the best individual rank, then by chunk ID, so the
        output is deterministic.
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    chunk_by_id: dict[str, Chunk] = {}

    for results in ranked_lists:
        for rank, result in enumerate(results, start=1):
            cid = result.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            best_rank[cid] = min(best_rank.get(cid, rank), rank)
            chunk_by_id[cid] = result.chunk

    ordered = sorted(scores, key=lambda cid: (-scores[cid], best_rank[cid], cid))
    if top_n is not None:
        ordered = ordered[:top_n]
    return [
        SearchResult(chunk=chunk_by_id[cid], score=scores[cid], rank=rank, method="hybrid")
        for rank, cid in enumerate(ordered, start=1)
    ]


class HybridRetriever:
    """Retrieve chunks with any of the four modes, chosen by config.

    Example::

        retriever = HybridRetriever(store, BM25Index(store.chunks), embedder, cfg.retrieval)
        output = retriever.retrieve("What does the learning rate control?")
        for r in output.results:
            print(r.rank, r.chunk.citation, round(r.score, 3))
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        encoder: TextEncoder,
        config: RetrievalConfig | None = None,
    ) -> None:
        if len(vector_store) != len(bm25_index):
            raise ValueError(
                f"Vector store ({len(vector_store)}) and BM25 index ({len(bm25_index)}) "
                "must contain the same chunks"
            )
        self.vector_store = vector_store
        self.bm25 = bm25_index
        self.encoder = encoder
        self.config = config or RetrievalConfig()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        mode: str | None = None,
    ) -> RetrievalOutput:
        """Run retrieval for one query.

        Args:
            query: The user's question.
            top_k: Override ``config.top_k``.
            mode: Override ``config.mode`` (heap, faiss, bm25, hybrid). The
                eval script uses this to compare modes on one index.

        Returns:
            A :class:`RetrievalOutput`. Empty results for an empty query or index.
        """
        mode = mode or self.config.mode
        k = top_k or self.config.top_k
        if mode not in ("heap", "faiss", "bm25", "hybrid"):
            raise ValueError(f"Unknown retrieval mode: {mode!r}")

        query = query.strip()
        if not query or len(self.vector_store) == 0:
            return RetrievalOutput(mode=mode)

        # Dense search is needed for every mode: for results (heap/faiss/hybrid)
        # and always for the refusal signal (best cosine similarity).
        dense_backend = mode if mode in ("heap", "faiss") else self.config.dense_backend
        dense_k = max(k, self.config.candidate_k) if mode == "hybrid" else max(k, 1)
        query_vector = self.encoder.encode_query(query)
        dense = self.vector_store.search(query_vector, dense_k, backend=dense_backend)
        max_similarity = dense[0].score if dense else None

        if mode in ("heap", "faiss"):
            results = dense[:k]
        elif mode == "bm25":
            results = self.bm25.search(query, k)
        else:
            sparse = self.bm25.search(query, max(k, self.config.candidate_k))
            results = reciprocal_rank_fusion([dense, sparse], k=self.config.rrf_k, top_n=k)

        logger.debug(
            "retrieve mode=%s k=%d -> %d results (max_sim=%s)",
            mode,
            k,
            len(results),
            f"{max_similarity:.3f}" if max_similarity is not None else "n/a",
        )
        return RetrievalOutput(results=results, max_similarity=max_similarity, mode=mode)


def build_demo_retriever(config) -> "HybridRetriever | None":
    """Index every PDF in ``config.paths.data_dir`` (used by the developer CLIs).

    The app and eval script use :class:`~notemind.pipeline.RAGPipeline`
    instead, which adds caching; this helper keeps the CLIs self-contained.
    """
    from notemind.chunker import Chunker
    from notemind.embedder import Embedder
    from notemind.pdf_loader import PDFLoader, PDFLoadError

    pdfs = sorted(config.paths.data_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {config.paths.data_dir}")
        return None

    pages = []
    for pdf in pdfs:
        try:
            pages.extend(PDFLoader().load(pdf).text_pages)
        except PDFLoadError as exc:
            print(f"Skipping {pdf.name}: {exc}")
    chunks = Chunker(config.chunking).chunk_pages(pages)
    if not chunks:
        print("No text chunks were produced from the PDFs.")
        return None

    embedder = Embedder(config.embedding)
    _ = embedder.model  # load (and on first run, download) before timing indexing
    started = time.perf_counter()
    vectors = embedder.encode([c.text for c in chunks])
    print(f"\nIndexed {len(chunks)} chunks from {len(pdfs)} PDF(s) "
          f"in {time.perf_counter() - started:.2f}s")

    store = VectorStore(embedder.dimension)
    store.add(chunks, vectors)
    return HybridRetriever(store, BM25Index(chunks), embedder, config.retrieval)


def _main(argv: list[str]) -> int:
    """Developer CLI: index the PDFs in data_dir and compare all four modes.

    Usage: python -m notemind.hybrid "your question"
    """
    from notemind.config import load_config, setup_logging

    if len(argv) != 1:
        print('Usage: python -m notemind.hybrid "your question"')
        return 2

    config = load_config()
    setup_logging(config.log_level)
    retriever = build_demo_retriever(config)
    if retriever is None:
        return 1

    print(f"Query: {argv[0]!r}   (min_similarity threshold = {config.retrieval.min_similarity})")
    for mode in ("heap", "faiss", "bm25", "hybrid"):
        started = time.perf_counter()
        output = retriever.retrieve(argv[0], mode=mode)
        ms = (time.perf_counter() - started) * 1000
        print(f"\n== {mode} ({ms:.1f} ms, best cosine {output.max_similarity:.3f})")
        if not output.results:
            print("   (no results)")
        for r in output.results:
            preview = r.chunk.text[:90].replace("\n", " ")
            print(f"  {r.rank}. {r.chunk.citation:<28} score={r.score:.4f}  {preview}...")
    if output.max_similarity is not None and output.max_similarity < config.retrieval.min_similarity:
        print("\nBest cosine is below min_similarity: the app would answer "
              "\"I couldn't find this in your notes\".")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
