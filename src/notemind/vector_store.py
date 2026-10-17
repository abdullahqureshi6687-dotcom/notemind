"""Dense vector storage with two interchangeable search backends.

* ``heap``  - the from-scratch O(n log k) top-k in :mod:`notemind.heap_topk`.
* ``faiss`` - FAISS ``IndexFlatIP``: exact inner-product search in optimised C++.

Both are *exact* searches over the same normalised vectors, so they must
return the same chunks (the tests check this). FAISS is used for speed; the
heap version exists to demonstrate the algorithm and act as a reference.
For millions of vectors you would switch to an approximate FAISS index such as
IVF or HNSW, trading a little recall for much faster search.

The store can be saved to and loaded from a folder:

    manifest.json   metadata (model name, dimension, chunk settings, files)
    chunks.json     the chunks, in row order
    vectors.npy     float32 matrix, one row per chunk (used by heap search)
    index.faiss     serialised FAISS index
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from notemind.embedder import l2_normalize
from notemind.heap_topk import cosine_top_k
from notemind.models import Chunk, SearchResult

logger = logging.getLogger(__name__)

INDEX_FORMAT_VERSION: int = 1
_MANIFEST = "manifest.json"
_CHUNKS = "chunks.json"
_VECTORS = "vectors.npy"
_FAISS = "index.faiss"


class IndexLoadError(Exception):
    """Raised when a saved index is missing, incomplete or inconsistent."""


class VectorStore:
    """Holds chunk embeddings and answers nearest-neighbour queries.

    Example::

        store = VectorStore(dimension=384)
        store.add(chunks, embedder.encode([c.text for c in chunks]))
        results = store.search(embedder.encode_query("what is a heap?"), k=5)
    """

    BACKENDS: tuple[str, ...] = ("heap", "faiss")

    def __init__(self, dimension: int) -> None:
        if dimension < 1:
            raise ValueError("dimension must be >= 1")
        self.dimension = dimension
        self.chunks: list[Chunk] = []
        self._vectors = np.zeros((0, dimension), dtype=np.float32)
        self._faiss = faiss.IndexFlatIP(dimension)

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def vectors(self) -> np.ndarray:
        """Read-only view of the stored (normalised) vectors."""
        view = self._vectors.view()
        view.flags.writeable = False
        return view

    @property
    def sources(self) -> list[str]:
        """Distinct PDF file names in the store, in insertion order."""
        return list(dict.fromkeys(chunk.source for chunk in self.chunks))

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        """Add chunks and their embeddings (row i belongs to chunk i).

        Raises:
            ValueError: On a count or dimension mismatch, or duplicate chunk IDs.
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(
                f"vectors must have shape (n, {self.dimension}), got {vectors.shape}"
            )
        if len(chunks) != vectors.shape[0]:
            raise ValueError(f"{len(chunks)} chunks but {vectors.shape[0]} vectors")
        if not chunks:
            return

        existing = {chunk.chunk_id for chunk in self.chunks}
        duplicates = [c.chunk_id for c in chunks if c.chunk_id in existing]
        if duplicates or len({c.chunk_id for c in chunks}) != len(chunks):
            raise ValueError(f"Duplicate chunk IDs, e.g. {duplicates[:3] or 'within batch'}")

        vectors = np.ascontiguousarray(l2_normalize(vectors))
        self.chunks.extend(chunks)
        self._vectors = np.vstack([self._vectors, vectors])
        self._faiss.add(vectors)
        logger.debug("VectorStore now holds %d chunks", len(self.chunks))

    def search(self, query: np.ndarray, k: int, backend: str = "faiss") -> list[SearchResult]:
        """Return the k chunks most similar to ``query`` (cosine similarity).

        Args:
            query: Query embedding, shape ``(dimension,)``.
            k: Number of results; capped at the number of stored chunks.
            backend: ``"faiss"`` or ``"heap"``.

        Returns:
            Results sorted by similarity, best first. Empty if the store is empty.
        """
        if backend not in self.BACKENDS:
            raise ValueError(f"backend must be one of {self.BACKENDS}, got {backend!r}")
        if len(self) == 0 or k <= 0:
            return []

        q = l2_normalize(np.asarray(query, dtype=np.float32).reshape(-1))
        if q.shape[0] != self.dimension:
            raise ValueError(f"query has dimension {q.shape[0]}, expected {self.dimension}")
        k = min(k, len(self))

        if backend == "heap":
            hits = cosine_top_k(q, self._vectors, k, assume_normalized=True)
        else:
            scores, ids = self._faiss.search(q.reshape(1, -1), k)
            hits = [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

        return [
            SearchResult(chunk=self.chunks[index], score=score, rank=rank, method=backend)
            for rank, (index, score) in enumerate(hits, start=1)
        ]

    def save(self, directory: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        """Save the store to ``directory`` (created if needed).

        The manifest is written last, so a crash mid-save leaves a folder that
        :meth:`load` rejects instead of a silently corrupt index.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / _MANIFEST).unlink(missing_ok=True)

        with open(directory / _CHUNKS, "w", encoding="utf-8") as fh:
            json.dump([c.to_dict() for c in self.chunks], fh, ensure_ascii=False)
        np.save(directory / _VECTORS, self._vectors)
        # serialize_index + Python file I/O instead of faiss.write_index:
        # write_index can fail on Windows paths containing non-ASCII characters.
        faiss.serialize_index(self._faiss).tofile(str(directory / _FAISS))

        manifest = {
            "format_version": INDEX_FORMAT_VERSION,
            "dimension": self.dimension,
            "num_chunks": len(self),
            "sources": self.sources,
            "metadata": metadata or {},
        }
        with open(directory / _MANIFEST, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False)
        logger.info("Saved index with %d chunks to %s", len(self), directory)
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> tuple["VectorStore", dict[str, Any]]:
        """Load a store saved with :meth:`save`.

        Returns:
            ``(store, metadata)`` where metadata is what was passed to ``save``.

        Raises:
            IndexLoadError: If files are missing or don't agree with each other.
        """
        directory = Path(directory)
        manifest_path = directory / _MANIFEST
        if not manifest_path.is_file():
            raise IndexLoadError(f"No saved index in {directory} (manifest.json missing)")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("format_version") != INDEX_FORMAT_VERSION:
                raise IndexLoadError(
                    f"Index format {manifest.get('format_version')} is not supported; rebuild it"
                )
            chunks_data = json.loads((directory / _CHUNKS).read_text(encoding="utf-8"))
            vectors = np.load(directory / _VECTORS)
            index = faiss.deserialize_index(np.fromfile(str(directory / _FAISS), dtype=np.uint8))
        except IndexLoadError:
            raise
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            raise IndexLoadError(f"Could not read index in {directory}: {exc}") from exc

        chunks = [Chunk.from_dict(item) for item in chunks_data]
        dimension = int(manifest["dimension"])
        expected = int(manifest["num_chunks"])
        if not (len(chunks) == vectors.shape[0] == index.ntotal == expected):
            raise IndexLoadError(
                f"Index files disagree: {len(chunks)} chunks, {vectors.shape[0]} vectors, "
                f"{index.ntotal} FAISS entries, manifest says {expected}. Rebuild the index."
            )
        if vectors.ndim != 2 or vectors.shape[1] != dimension or index.d != dimension:
            raise IndexLoadError("Index dimension mismatch. Rebuild the index.")

        store = cls(dimension)
        store.chunks = chunks
        store._vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        store._faiss = index
        logger.info("Loaded index with %d chunks from %s", len(store), directory)
        return store, dict(manifest.get("metadata", {}))
