"""Sentence embeddings with sentence-transformers (all-MiniLM-L6-v2 by default).

The model is loaded lazily on first use, so importing this module is cheap and
code paths that never embed (e.g. BM25-only evaluation) never pay the cost.
In the Streamlit app the Embedder instance is wrapped in ``st.cache_resource``
so the model loads once per server process, not on every rerun.

All vectors are L2-normalised. That makes cosine similarity equal to the dot
product, which is what both the heap search and FAISS ``IndexFlatIP`` compute.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from notemind.config import EmbeddingConfig

logger = logging.getLogger(__name__)


@runtime_checkable
class TextEncoder(Protocol):
    """Anything that turns text into unit-length vectors.

    The real :class:`Embedder` satisfies this, and so does the small fake
    encoder used in the tests, which lets the retrieval code be tested
    without downloading a model.
    """

    model_name: str

    @property
    def dimension(self) -> int: ...

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Return a float32 copy with every row scaled to unit length.

    Zero rows stay zero instead of becoming NaN.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        norm = float(np.linalg.norm(vectors))
        return vectors / norm if norm > 0 else vectors.copy()
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


class Embedder:
    """Wrapper around a SentenceTransformer model.

    Example::

        embedder = Embedder(config.embedding)
        doc_vectors = embedder.encode([c.text for c in chunks])   # (n, 384)
        query_vector = embedder.encode_query("What is backprop?")  # (384,)
    """

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig()
        self.model_name: str = self.config.model_name
        self._model = None  # loaded on first use

    @property
    def model(self):  # -> SentenceTransformer (imported lazily)
        """The underlying SentenceTransformer, loaded on first access."""
        if self._model is None:
            # Imported here so that `import notemind.embedder` does not load torch.
            from sentence_transformers import SentenceTransformer

            started = time.perf_counter()
            logger.info("Loading embedding model %s on %s ...", self.model_name, self.config.device)
            self._model = SentenceTransformer(self.model_name, device=self.config.device)
            logger.info(
                "Model loaded in %.1fs (dimension=%d, max_seq_length=%d tokens)",
                time.perf_counter() - started,
                self.dimension,
                self.max_seq_length,
            )
        return self._model

    @property
    def dimension(self) -> int:
        """Embedding size (384 for all-MiniLM-L6-v2)."""
        model = self.model
        getter = getattr(model, "get_embedding_dimension", None) or getattr(
            model, "get_sentence_embedding_dimension"
        )
        return int(getter())

    @property
    def max_seq_length(self) -> int:
        """Token limit; text beyond this is truncated before embedding."""
        return int(getattr(self.model, "max_seq_length", 0) or 0)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Embed many texts.

        Args:
            texts: Chunk texts. Empty input returns an array of shape (0, d).

        Returns:
            Float32 array of shape ``(len(texts), dimension)``, rows unit length.
        """
        if len(texts) == 0:
            return np.zeros((0, self.dimension), dtype=np.float32)
        model = self.model  # load first, so the timing below measures only encoding
        started = time.perf_counter()
        # Passed positionally: sentence-transformers 6.x renamed this argument.
        vectors = model.encode(
            list(texts),
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors = l2_normalize(vectors)  # cheap safety net; keeps dtype float32
        logger.log(
            logging.INFO if len(texts) > 1 else logging.DEBUG,  # don't log every query
            "Embedded %d texts in %.2fs",
            len(texts),
            time.perf_counter() - started,
        )
        return vectors

    def encode_query(self, text: str) -> np.ndarray:
        """Embed a single query. Returns shape ``(dimension,)``."""
        return self.encode([text])[0]
