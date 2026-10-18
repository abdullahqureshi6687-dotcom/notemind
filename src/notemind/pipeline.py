"""RAGPipeline: the one object the UI and the eval script talk to.

It owns the documents, builds and caches the index, and wires together
loading -> chunking -> embedding -> retrieval -> generation:

    pipeline = RAGPipeline(config)
    pipeline.add_pdf_path("data/sample_pdfs/lecture1.pdf")
    pipeline.build_index()
    answer = pipeline.ask("What is gradient descent?")

**Index caching.** Embedding is the slow step, so built indexes are saved to
``indexes/<key>/``. The key is a SHA-256 hash of everything that affects the
vectors: the embedding model, the chunking settings, the index format version,
and the content hash of every PDF. Any change produces a new key, so a stale
index can never be loaded by mistake, and re-uploading the same files with the
same settings loads in milliseconds instead of re-embedding.

**Dependency injection.** The embedder and LLM client can be passed in. The
Streamlit app passes one cached embedder shared by all sessions; the tests
pass fakes so they run without a model download or an API key.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from notemind.bm25_index import BM25Index
from notemind.chunker import Chunker
from notemind.config import AppConfig
from notemind.embedder import Embedder, TextEncoder
from notemind.generator import Generator
from notemind.hybrid import HybridRetriever, RetrievalOutput
from notemind.llm_client import LLMClient, create_llm_client
from notemind.models import Answer, Chunk, LoadReport, Page
from notemind.pdf_loader import PDFLoader
from notemind.prompts import REFUSAL_TEXT
from notemind.quiz import Quiz, QuizGenerator, filter_pages
from notemind.vector_store import INDEX_FORMAT_VERSION, IndexLoadError, VectorStore

logger = logging.getLogger(__name__)

NO_INDEX_MESSAGE = "Upload at least one PDF with selectable text to start asking questions."


@dataclass
class DocumentInfo:
    """What the UI shows for each loaded PDF."""

    source: str
    num_pages: int
    text_pages: int
    sha256: str
    report: LoadReport


@dataclass
class IndexStats:
    """Summary of the last index build, for logging and the UI."""

    num_chunks: int
    num_documents: int
    from_cache: bool
    seconds: float
    key: str


class RAGPipeline:
    """End-to-end retrieval-augmented generation over a set of PDFs."""

    def __init__(
        self,
        config: AppConfig,
        embedder: TextEncoder | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        """
        Args:
            config: Validated application config.
            embedder: Shared encoder; created lazily from config if omitted.
            llm: LLM client; created lazily (on first question) if omitted,
                so retrieval and evaluation work without an API key.
        """
        self.config = config
        self._embedder = embedder
        self._llm = llm
        self._api_key: str | None = None  # per-session key typed by the user (bring your own key)
        self._loader = PDFLoader()

        self._pages: dict[str, list[Page]] = {}  # source -> pages, insertion order
        self._docs: dict[str, DocumentInfo] = {}

        self._store: VectorStore | None = None
        self._bm25: BM25Index | None = None
        self._retriever: HybridRetriever | None = None
        self._index_key: str | None = None
        self.last_index_stats: IndexStats | None = None

    # ------------------------------------------------------------------ documents

    @property
    def embedder(self) -> TextEncoder:
        if self._embedder is None:
            self._embedder = Embedder(self.config.embedding)
        return self._embedder

    @property
    def documents(self) -> list[DocumentInfo]:
        """Loaded PDFs, in the order they were added."""
        return list(self._docs.values())

    @property
    def pages(self) -> list[Page]:
        """All pages of all loaded PDFs (including empty ones)."""
        return [page for pages in self._pages.values() for page in pages]

    def add_pdf_bytes(self, data: bytes, filename: str) -> LoadReport:
        """Load a PDF from bytes. Re-adding identical content is a no-op.

        A different file with the same name replaces the old one.

        Raises:
            PDFLoadError: If the bytes are not a readable PDF.
        """
        digest = hashlib.sha256(data).hexdigest()
        source = Path(filename).name
        existing = self._docs.get(source)
        if existing and existing.sha256 == digest:
            return existing.report

        result = self._loader.load_bytes(data, source)
        self._pages[source] = result.pages
        self._docs[source] = DocumentInfo(
            source=source,
            num_pages=result.report.num_pages,
            text_pages=result.report.text_pages,
            sha256=digest,
            report=result.report,
        )
        self._invalidate()
        return result.report

    def add_pdf_path(self, path: str | Path) -> LoadReport:
        """Load a PDF from disk (see :meth:`add_pdf_bytes`)."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"PDF not found: {path}")
        return self.add_pdf_bytes(path.read_bytes(), path.name)

    def add_folder(self, folder: str | Path) -> list[LoadReport]:
        """Load every ``*.pdf`` in a folder (non-recursive), sorted by name."""
        return [self.add_pdf_path(p) for p in sorted(Path(folder).glob("*.pdf"))]

    def remove_document(self, source: str) -> None:
        """Forget a PDF. The index must be rebuilt afterwards."""
        if self._docs.pop(source, None) is not None:
            self._pages.pop(source, None)
            self._invalidate()

    def clear(self) -> None:
        """Remove all documents and the index."""
        self._pages.clear()
        self._docs.clear()
        self._invalidate()

    # ------------------------------------------------------------------ config

    def update_config(self, config: AppConfig) -> None:
        """Apply new settings, invalidating only what the change affects.

        * chunking or embedding changed -> the index must be rebuilt
        * retrieval changed             -> the retriever picks it up directly
        * generation changed            -> a new LLM client is created lazily
        """
        old = self.config
        self.config = config
        if config.chunking != old.chunking or config.embedding != old.embedding:
            if config.embedding != old.embedding:
                self._embedder = None
            self._invalidate()
        elif self._retriever is not None and config.retrieval != old.retrieval:
            self._retriever.config = config.retrieval
        if config.generation != old.generation:
            self._llm = None

    # ------------------------------------------------------------------ indexing

    @property
    def is_indexed(self) -> bool:
        """True if an up-to-date index exists for the current documents."""
        return self._retriever is not None

    @property
    def needs_indexing(self) -> bool:
        """True if there are documents with text but no current index."""
        return not self.is_indexed and any(d.text_pages for d in self._docs.values())

    @property
    def num_chunks(self) -> int:
        return len(self._store) if self._store is not None else 0

    def index_key(self) -> str:
        """Hash of everything that determines the index contents."""
        payload = {
            "format": INDEX_FORMAT_VERSION,
            "model": self.config.embedding.model_name,
            "chunk_size": self.config.chunking.chunk_size,
            "chunk_overlap": self.config.chunking.chunk_overlap,
            "min_chunk_chars": self.config.chunking.min_chunk_chars,
            "files": sorted((d.source, d.sha256) for d in self._docs.values()),
        }
        blob = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def build_index(self, use_cache: bool = True) -> IndexStats | None:
        """Chunk, embed and index all documents (or load them from cache).

        Returns:
            Build statistics, or None if there is no text to index.
        """
        started = time.perf_counter()
        chunks = Chunker(self.config.chunking).chunk_pages(self.pages)
        if not chunks:
            logger.warning("Nothing to index: no text chunks in the loaded documents")
            self._invalidate()
            return None

        key = self.index_key()
        cache_dir = self.config.paths.index_dir / key[:16]
        store = self._load_cached(cache_dir, key, chunks) if use_cache else None
        from_cache = store is not None

        if store is None:
            vectors = self.embedder.encode([c.text for c in chunks])
            store = VectorStore(vectors.shape[1])
            store.add(chunks, vectors)
            if use_cache:
                try:
                    store.save(cache_dir, metadata={"key": key, "model": self.embedder.model_name})
                except OSError as exc:  # read-only disk, full disk: caching is optional
                    logger.warning("Could not save index cache to %s: %s", cache_dir, exc)

        self._set_index(store, key)
        stats = IndexStats(
            num_chunks=len(store),
            num_documents=len(self._docs),
            from_cache=from_cache,
            seconds=time.perf_counter() - started,
            key=key[:16],
        )
        self.last_index_stats = stats
        logger.info(
            "Index ready: %d chunks from %d document(s) in %.2fs (%s)",
            stats.num_chunks,
            stats.num_documents,
            stats.seconds,
            "cache" if from_cache else "built",
        )
        return stats

    def _load_cached(self, cache_dir: Path, key: str, chunks: list[Chunk]) -> VectorStore | None:
        """Load a cached index if it matches the current key and chunks exactly."""
        if not (cache_dir / "manifest.json").is_file():
            return None
        try:
            store, metadata = VectorStore.load(cache_dir)
        except IndexLoadError as exc:
            logger.warning("Ignoring unusable cached index: %s", exc)
            return None
        if metadata.get("key") != key or [c.chunk_id for c in store.chunks] != [c.chunk_id for c in chunks]:
            logger.warning("Cached index does not match current documents; rebuilding")
            return None
        return store

    def _set_index(self, store: VectorStore, key: str) -> None:
        self._store = store
        self._bm25 = BM25Index(store.chunks)
        self._retriever = HybridRetriever(store, self._bm25, self.embedder, self.config.retrieval)
        self._index_key = key

    def _invalidate(self) -> None:
        self._store = self._bm25 = self._retriever = None
        self._index_key = None

    # ------------------------------------------------------------------ querying

    @property
    def llm(self) -> LLMClient:
        """LLM client, created on first use. Raises LLMConfigError without a key."""
        if self._llm is None:
            self._llm = create_llm_client(self.config.generation, api_key=self._api_key)
        return self._llm

    def set_api_key(self, api_key: str | None) -> None:
        """Use a caller-supplied API key instead of the environment (None = environment).

        The Streamlit app uses this for "bring your own key": the key lives only
        in this pipeline object, i.e. in one browser session's memory.
        """
        api_key = (api_key or "").strip() or None
        if api_key != self._api_key:
            self._api_key = api_key
            self._llm = None

    def retrieve(
        self, question: str, top_k: int | None = None, mode: str | None = None
    ) -> RetrievalOutput:
        """Retrieval only (no LLM). Used by the eval script and the UI."""
        if self._retriever is None:
            return RetrievalOutput(mode=mode or self.config.retrieval.mode)
        return self._retriever.retrieve(question, top_k=top_k, mode=mode)

    def ask(self, question: str, top_k: int | None = None, mode: str | None = None) -> Answer:
        """Answer a question from the notes.

        Refuses without calling the LLM when there is no index or when the best
        cosine similarity is below ``retrieval.min_similarity``. In that case
        the closest passages are still returned as ``sources`` so the user can
        see what was found.

        Raises:
            LLMError: If the provider call fails (the UI shows a friendly message).
        """
        if not question.strip():
            return Answer(text="Please type a question.", refused=True, refusal_reason="empty_question")
        if self._retriever is None:
            return Answer(text=NO_INDEX_MESSAGE, refused=True, refusal_reason="no_index")

        retrieval = self.retrieve(question, top_k=top_k, mode=mode)
        best = retrieval.max_similarity
        if best is None or best < self.config.retrieval.min_similarity:
            logger.info(
                "Refusing without LLM call: best cosine %s < %.2f",
                f"{best:.3f}" if best is not None else "n/a",
                self.config.retrieval.min_similarity,
            )
            return Answer(
                text=REFUSAL_TEXT,
                sources=retrieval.results,
                refused=True,
                refusal_reason="low_similarity",
                max_similarity=best,
            )

        answer = Generator(self.llm, self.config.generation).answer(question, retrieval.results)
        answer.max_similarity = best
        return answer

    def make_quiz(
        self,
        source: str,
        num_questions: int | None = None,
        start_page: int | None = None,
        end_page: int | None = None,
        seed: int | None = None,
    ) -> Quiz:
        """Generate a quiz from one PDF, optionally limited to a page range.

        Raises:
            KeyError: If ``source`` is not a loaded document.
            QuizGenerationError, LLMError: See :class:`QuizGenerator`.
        """
        if source not in self._pages:
            raise KeyError(f"Unknown document: {source}")
        pages = filter_pages(self._pages[source], source, start_page, end_page)
        return QuizGenerator(self.llm, self.config.quiz, seed=seed).generate(pages, num_questions)
