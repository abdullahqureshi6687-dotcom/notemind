"""BM25 keyword search over chunks, built on rank-bm25.

Why keyword search in a semantic-search project? Embeddings are good at
meaning ("how do networks learn" ~ "backpropagation") but weak at exact,
rare tokens: course codes, acronyms like "RRF" or "KNN", formula names,
variable names. BM25 scores documents by term frequency, weighted by how rare
each term is (IDF), with saturation and length normalisation.

Two details worth knowing:

1. **Candidate filtering with an inverted index.** Only chunks that contain at
   least one query term are scored and returned. Without this, a query whose
   words appear nowhere would still "return" arbitrary zero-score chunks.
2. **The negative-IDF quirk.** In BM25Okapi, a term that appears in more than
   half of all chunks gets a negative IDF (rank-bm25 replaces it with a small
   epsilon value, which can still be negative in tiny corpora). So a matching
   chunk can score <= 0. That is why results are ranked by score but filtered
   by term overlap, never by ``score > 0``.

The index is not saved to disk: rebuilding it from chunk texts takes
milliseconds, so it is rebuilt whenever a saved vector index is loaded.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from rank_bm25 import BM25Okapi

from notemind.heap_topk import top_k
from notemind.models import Chunk, SearchResult

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")

# Small English stopword list; deliberately conservative so that short
# technical terms ("a*", "k", "no") in questions are not over-filtered.
STOPWORDS: frozenset[str] = frozenset(
    """a an and are as at be by for from has have how i in is it its of on or
    that the this to was were what when where which who why will with do does
    did can could should would you your we our they their there these those
    about into than then so if not but also such""".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase, split into word tokens, drop stopwords.

    Joined tokens such as ``"gpt-4"``, ``"3.5"`` and ``"k-means"`` are kept
    whole *and* also emitted as their parts, so a query for "k means" still
    matches notes that write "k-means".
    """
    tokens: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        if tok not in STOPWORDS:
            tokens.append(tok)
        if any(sep in tok for sep in "._-"):
            tokens.extend(p for p in re.split(r"[._-]", tok) if p and p not in STOPWORDS)
    return tokens


class BM25Index:
    """Keyword index over a list of chunks.

    Example::

        bm25 = BM25Index(chunks)
        results = bm25.search("chain rule backpropagation", k=5)
    """

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        """
        Args:
            chunks: Chunks to index (order defines internal IDs).
            k1: Term-frequency saturation. Higher = repeated terms count more.
            b: Length normalisation. 1.0 = fully normalise by chunk length.
        """
        self.chunks: list[Chunk] = list(chunks)
        self.k1, self.b = k1, b
        self._tokenized: list[list[str]] = [tokenize(c.text) for c in self.chunks]

        # Inverted index: token -> set of chunk positions containing it.
        self._postings: dict[str, set[int]] = {}
        for position, tokens in enumerate(self._tokenized):
            for token in set(tokens):
                self._postings.setdefault(token, set()).add(position)

        # rank-bm25 divides by the corpus size, so it can't be built empty.
        has_tokens = any(self._tokenized)
        self._bm25 = BM25Okapi(self._tokenized, k1=k1, b=b) if has_tokens else None
        logger.debug("BM25 index: %d chunks, %d unique terms", len(self.chunks), len(self._postings))

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def vocabulary_size(self) -> int:
        return len(self._postings)

    def search(self, query: str, k: int) -> list[SearchResult]:
        """Return up to k chunks ranked by BM25 score.

        Only chunks sharing at least one (non-stopword) term with the query
        are returned, so the result can be shorter than k, or empty.
        """
        if k <= 0 or self._bm25 is None:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        candidates: set[int] = set()
        for token in set(query_tokens):
            candidates |= self._postings.get(token, set())
        if not candidates:
            return []

        all_scores = self._bm25.get_scores(query_tokens)
        ordered = sorted(candidates)  # deterministic tie-breaking by position
        hits = top_k((all_scores[i] for i in ordered), k)
        return [
            SearchResult(chunk=self.chunks[ordered[j]], score=float(score), rank=rank, method="bm25")
            for rank, (j, score) in enumerate(hits, start=1)
        ]
