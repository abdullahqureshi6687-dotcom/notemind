"""Tests for the from-scratch MinHeap, top_k and cosine_top_k."""

from __future__ import annotations

import random

import numpy as np
import pytest

from notemind.heap_topk import MinHeap, cosine_similarity, cosine_top_k, top_k


# ----------------------------------------------------------------- MinHeap


@pytest.mark.parametrize("seed", range(10))
def test_heap_pops_in_ascending_order(seed: int) -> None:
    rng = random.Random(seed)
    values = [rng.randint(-100, 100) for _ in range(rng.randint(1, 200))]
    heap: MinHeap[int] = MinHeap()
    for v in values:
        heap.push(v, v)
    popped = [heap.pop()[0] for _ in range(len(values))]
    assert popped == sorted(values)
    assert len(heap) == 0


def test_heap_peek_returns_minimum_without_removing() -> None:
    heap: MinHeap[str] = MinHeap()
    for key in (5, 3, 8, 1):
        heap.push(key, f"item{key}")
    assert heap.peek() == (1, "item1")
    assert len(heap) == 4


def test_replace_root_keeps_heap_property() -> None:
    heap: MinHeap[int] = MinHeap()
    for v in (4, 7, 9, 10):
        heap.push(v, v)
    assert heap.replace_root(8, 8)[0] == 4
    assert [heap.pop()[0] for _ in range(4)] == [7, 8, 9, 10]


@pytest.mark.parametrize("method", ["pop", "peek"])
def test_empty_heap_raises(method: str) -> None:
    with pytest.raises(IndexError):
        getattr(MinHeap(), method)()


# ----------------------------------------------------------------- top_k


@pytest.mark.parametrize("seed", range(30))
def test_top_k_matches_numpy_including_ties(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 300))
    k = int(rng.integers(1, 25))
    scores = np.round(rng.random(n), 1)  # rounding creates many ties
    expected = list(np.argsort(-scores, kind="stable")[:k])
    assert [i for i, _ in top_k(scores, k)] == expected


def test_top_k_returns_scores_best_first() -> None:
    result = top_k([0.2, 0.9, 0.5, 0.7], 3)
    assert result == [(1, 0.9), (3, 0.7), (2, 0.5)]


def test_top_k_tie_break_prefers_lower_index() -> None:
    assert [i for i, _ in top_k([0.5, 0.5, 0.5, 0.1], 2)] == [0, 1]


@pytest.mark.parametrize("k", [0, -3])
def test_top_k_non_positive_k_returns_empty(k: int) -> None:
    assert top_k([1.0, 2.0], k) == []


def test_top_k_with_k_larger_than_input_returns_everything_sorted() -> None:
    assert [i for i, _ in top_k([3.0, 1.0, 2.0], 10)] == [0, 2, 1]


def test_top_k_handles_empty_input_generators_negatives_and_nan() -> None:
    assert top_k([], 3) == []
    assert top_k((x for x in [-1.0, -0.5, -2.0]), 2) == [(1, -0.5), (0, -1.0)]
    assert top_k([0.1, float("nan"), 0.9], 3) == [(2, 0.9), (0, 0.1)]


# ----------------------------------------------------------------- cosine


def test_cosine_similarity_known_values() -> None:
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([1, 2], [-1, -2]) == pytest.approx(-1.0)
    assert cosine_similarity([0, 0], [1, 1]) == 0.0  # zero vector: defined as 0, not NaN


@pytest.mark.parametrize("seed", range(10))
def test_cosine_top_k_matches_brute_force(seed: int) -> None:
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((80, 16)).astype(np.float32)
    query = rng.standard_normal(16).astype(np.float32)
    brute = sorted(range(80), key=lambda i: -cosine_similarity(matrix[i], query))[:5]

    assert [i for i, _ in cosine_top_k(query, matrix, 5)] == brute

    unit = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    unit_query = query / np.linalg.norm(query)
    assert [i for i, _ in cosine_top_k(unit_query, unit, 5, assume_normalized=True)] == brute


def test_cosine_top_k_zero_rows_and_edge_cases() -> None:
    matrix = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    assert cosine_top_k(np.array([1.0, 0.0]), matrix, 2) == [(1, pytest.approx(1.0)), (0, 0.0)]
    assert cosine_top_k(np.array([1.0, 0.0]), np.zeros((0, 2), dtype=np.float32), 3) == []
    with pytest.raises(ValueError):
        cosine_top_k(np.array([1.0, 0.0, 0.0]), matrix, 1)
