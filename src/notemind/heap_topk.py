"""From-scratch top-k selection with a binary min-heap.

Why a heap? To find the k best of n scores you don't need to sort all n
(O(n log n)). Keep a min-heap of size k: its root is the *worst* of the
current best k. For each new score, if it beats the root, replace the root and
sift down. Each step costs O(log k), so the total is **O(n log k)** time and
**O(k)** extra memory. With n = 10,000 chunks and k = 5 that is a big saving.

The heap is implemented by hand (no ``heapq``) to show the data structure:
an array where the children of index i are at 2i+1 and 2i+2.

In this project the heap version is the *reference* implementation. FAISS is
faster in practice (vectorised C++), and the tests check that both return the
same results.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Generic, TypeVar

import numpy as np

T = TypeVar("T")


class MinHeap(Generic[T]):
    """Array-backed binary min-heap of ``(key, item)`` pairs.

    Keys must be mutually comparable (tuples work, which is how ties are
    broken deterministically in :func:`top_k`).
    """

    def __init__(self) -> None:
        self._data: list[tuple[object, T]] = []

    def __len__(self) -> int:
        return len(self._data)

    def __bool__(self) -> bool:
        return bool(self._data)

    def peek(self) -> tuple[object, T]:
        """Return the smallest ``(key, item)`` without removing it. O(1)."""
        if not self._data:
            raise IndexError("peek from an empty heap")
        return self._data[0]

    def push(self, key: object, item: T) -> None:
        """Insert an item. O(log n)."""
        self._data.append((key, item))
        self._sift_up(len(self._data) - 1)

    def pop(self) -> tuple[object, T]:
        """Remove and return the smallest ``(key, item)``. O(log n)."""
        if not self._data:
            raise IndexError("pop from an empty heap")
        last = self._data.pop()
        if not self._data:
            return last
        smallest = self._data[0]
        self._data[0] = last
        self._sift_down(0)
        return smallest

    def replace_root(self, key: object, item: T) -> tuple[object, T]:
        """Pop the smallest item and push a new one in a single O(log n) step."""
        if not self._data:
            raise IndexError("replace_root on an empty heap")
        smallest = self._data[0]
        self._data[0] = (key, item)
        self._sift_down(0)
        return smallest

    def items(self) -> list[tuple[object, T]]:
        """All ``(key, item)`` pairs in heap (not sorted) order."""
        return list(self._data)

    def _sift_up(self, index: int) -> None:
        """Move the element at ``index`` up until its parent is not larger."""
        data = self._data
        while index > 0:
            parent = (index - 1) // 2
            if data[index][0] < data[parent][0]:
                data[index], data[parent] = data[parent], data[index]
                index = parent
            else:
                break

    def _sift_down(self, index: int) -> None:
        """Move the element at ``index`` down until both children are larger."""
        data = self._data
        size = len(data)
        while True:
            left, right = 2 * index + 1, 2 * index + 2
            smallest = index
            if left < size and data[left][0] < data[smallest][0]:
                smallest = left
            if right < size and data[right][0] < data[smallest][0]:
                smallest = right
            if smallest == index:
                return
            data[index], data[smallest] = data[smallest], data[index]
            index = smallest


def top_k(scores: Iterable[float], k: int) -> list[tuple[int, float]]:
    """Indices and values of the k largest scores, best first.

    Ties are broken by index (lower index ranks higher), so results are
    deterministic and match ``np.argsort(-scores, kind="stable")``.

    Args:
        scores: Any iterable of numbers (list, generator, NumPy array).
        k: Number of results wanted. ``k <= 0`` returns an empty list.

    Returns:
        Up to k ``(index, score)`` pairs sorted by score descending.

    Complexity:
        O(n log k) time, O(k) memory.
    """
    if k <= 0:
        return []

    heap: MinHeap[int] = MinHeap()
    for index, score in enumerate(scores):
        value = float(score)
        if value != value:  # NaN never ranks (NaN != NaN)
            continue
        # Key (score, -index): among equal scores, the higher index is
        # "smaller", so it is evicted first and the lower index survives.
        key = (value, -index)
        if len(heap) < k:
            heap.push(key, index)
        elif key > heap.peek()[0]:
            heap.replace_root(key, index)

    ranked = sorted(heap.items(), key=lambda pair: pair[0], reverse=True)
    return [(index, key[0]) for key, index in ranked]  # type: ignore[index]


def cosine_similarity(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray) -> float:
    """Cosine similarity of two vectors: ``a·b / (|a| |b|)``. Returns 0.0 for zero vectors."""
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    norm = float(np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
    if norm == 0.0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / norm)


def cosine_top_k(
    query: np.ndarray,
    matrix: np.ndarray,
    k: int,
    assume_normalized: bool = False,
) -> list[tuple[int, float]]:
    """Top-k rows of ``matrix`` by cosine similarity to ``query``.

    Similarities are computed with one matrix-vector product (vectorised), and
    the selection is done with the hand-written heap above. If the vectors are
    already L2-normalised (as the Embedder produces), cosine similarity is just
    the dot product, which ``assume_normalized=True`` skips re-normalising.

    Args:
        query: Shape ``(d,)``.
        matrix: Shape ``(n, d)``, one embedding per row.
        k: Number of results.
        assume_normalized: Skip normalisation when inputs are unit vectors.

    Returns:
        Up to k ``(row_index, cosine_similarity)`` pairs, best first.
    """
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got shape {matrix.shape}")
    if matrix.shape[0] == 0 or k <= 0:
        return []
    query = np.asarray(query, dtype=np.float32).reshape(-1)
    if query.shape[0] != matrix.shape[1]:
        raise ValueError(f"query dim {query.shape[0]} != matrix dim {matrix.shape[1]}")

    if assume_normalized:
        scores = matrix @ query
    else:
        row_norms = np.linalg.norm(matrix, axis=1)
        query_norm = float(np.linalg.norm(query))
        denom = row_norms * query_norm
        with np.errstate(divide="ignore", invalid="ignore"):
            scores = np.where(denom > 0, (matrix @ query) / denom, 0.0)
    return top_k(scores, k)
